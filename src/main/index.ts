/**
 * Main process entry point.
 *
 * Responsibilities kept here: window and tray lifecycle, first-run relocation, wiring the
 * bridge to IPC, and orderly shutdown. Everything else lives in its own module.
 */

import { app, BrowserWindow, ipcMain, Menu, Tray, dialog, nativeImage, shell } from 'electron'
import path from 'node:path'
import { ensureDirs } from './storage/paths'
import { loadSettings, patchSettings } from './storage/settings'
import { getDb } from './storage/db'
import { detectHardware } from './hardware/gpu'
import { scanLibrary } from './models/library'
import { checkRelocation } from './models/relocation'
import { downloadQueue } from './downloads/queue'
import { mcpManager } from './agent/mcp'
import { getHfToken } from './remote/auth'
import { apiServer } from './api/server'
import { remoteWeb } from './remote/web'
import { llama } from './runtime/llama'
import { vendorDiagnostics } from './runtime/binaries'
import { handlers, invokeBridge, setEmitter, setLibrary, shutdown, modelsDir, startHardwareRefresh, stopHardwareRefresh } from './bridge'
import { installCrashHandlers, logger } from './log'

let mainWindow: BrowserWindow | null = null
let tray: Tray | null = null
/**
 * The user has already said they want to quit, so the close prompt must not appear again.
 *
 * Deliberately separate from `shuttingDown`. One flag used to serve both purposes, and the two
 * questions have different answers: the tray's Quit and the close dialog's Quit both set it
 * before calling app.quit(), which then made `before-quit` believe shutdown had already run and
 * return immediately. Those are the two most likely ways to quit, and both left llama-server
 * holding VRAM, MCP servers running, and background jobs alive after the app was gone.
 */
let userChoseQuit = false
/** Set once the shutdown handler has taken responsibility for this quit. */
let shuttingDown = false

/**
 * One emitter feeds both surfaces: the desktop window over IPC, and every connected remote
 * browser over SSE. Neither can silently miss an event the other gets.
 *
 * Named rather than inlined into `setEmitter` so the tray — which is not a bridge caller — can
 * announce things through the same path.
 */
function emitToSurfaces(channel: string, payload: unknown): void {
  if (!mainWindow?.isDestroyed()) mainWindow?.webContents.send(channel, payload)
  remoteWeb.broadcast(channel, payload)
}

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1040,
    minHeight: 660,
    show: false,
    // Matches --bg in styles.css, so the window does not flash the old cool-grey before paint.
    backgroundColor: '#1f1e1d',
    title: 'LLM Manager',
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  })

  /*
   * Under test, appear without taking focus.
   *
   * The end-to-end suite runs several app instances at once, and `show()` pulls focus to each as
   * it opens — six windows grabbing the keyboard in turn, off whatever the person at the machine
   * was actually doing. Playwright drives the page over the debugging protocol rather than
   * through the OS, so nothing it does needs the window focused.
   */
  mainWindow.on('ready-to-show', () =>
    process.env.LLMM_E2E ? mainWindow?.showInactive() : mainWindow?.show()
  )

  // Round 11: ask on first close, then remember the answer.
  mainWindow.on('close', (event) => {
    if (userChoseQuit) return
    const settings = loadSettings()

    if (settings.ui.closeAction === 'quit') return
    if (settings.ui.closeAction === 'tray') {
      event.preventDefault()
      mainWindow?.hide()
      return
    }

    event.preventDefault()
    const choice = dialog.showMessageBoxSync(mainWindow!, {
      type: 'question',
      buttons: ['Minimise to tray', 'Quit'],
      defaultId: 0,
      cancelId: 0,
      title: 'Close LLM Manager',
      message: 'Keep running in the background?',
      detail:
        'Minimising to the tray keeps the loaded model in memory and the API server available. ' +
        'Quitting stops both, and disconnects anyone using the remote web UI.\n\n' +
        'This choice is remembered; change it in Settings.'
    })

    const action = choice === 1 ? 'quit' : 'tray'
    patchSettings({ ui: { closeAction: action } })

    if (action === 'quit') {
      userChoseQuit = true
      app.quit()
    } else {
      mainWindow?.hide()
    }
  })

  if (process.env['ELECTRON_RENDERER_URL']) {
    void mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    void mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'))
  }

  // External links open in the real browser, never inside the app window.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url)
    return { action: 'deny' }
  })
}

function createTray(): void {
  /*
   * The tray mark, inlined so it needs nothing from the package layout.
   *
   * This was a 1x1 transparent PNG, with a note saying a real one would arrive at branding
   * time. Minimising to the tray is offered in the close dialog, so the placeholder meant the
   * app could hide itself behind an icon nobody could see, find or click. Generated by
   * scripts/make-icon.mjs from the same mark the sidebar draws.
   */
  const icon = nativeImage.createFromDataURL(
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAqklEQVR42u1XwQnAIAx0hIzmz68j+HMcR3CeTOEIFiGPIlihtl4oDdxPyXlnTDTmj4Xg6GwDkkBqQCUnjq4ICEHAc3RV4BEE8olARshfOxBK/v02dPLvtWEg/z4bOLpwQSC8ffIgdT8iUGQNPZnUDzyfIcte2pn0PhlpLGki8SqK5LD6CMAtUHMJVZWhqodIzVMMb0Za2jF2IIGPZFqGUuxYDv+YqPiafSIOJymIR4+NvWcAAAAASUVORK5CYII='
  )
  tray = new Tray(icon)
  tray.setToolTip('LLM Manager')

  const rebuild = (): void => {
    const loaded = llama.loaded
    tray?.setContextMenu(
      Menu.buildFromTemplate([
        { label: loaded ? `Loaded: ${loaded.model.filename}` : 'No model loaded', enabled: false },
        { label: apiServer.running ? `API on port ${apiServer.port}` : 'API server stopped', enabled: false },
        { type: 'separator' },
        {
          label: 'Show window',
          click: () => {
            mainWindow?.show()
            mainWindow?.focus()
          }
        },
        {
          label: 'Unload model',
          enabled: !!loaded,
          // Through the bridge, not straight to the runtime. Calling `llama.unload()` here
          // skipped the `model:status` the handler emits, so the window went on showing the
          // model as loaded — with a working composer that answered "No model is loaded".
          click: () => void invokeBridge('model:unload', [])
        },
        { type: 'separator' },
        {
          label: 'Quit',
          click: () => {
            userChoseQuit = true
            app.quit()
          }
        }
      ])
    )
  }

  rebuild()
  llama.on('status', rebuild)

  /*
   * A backend that dies takes the "loaded" badge with it.
   *
   * llama-server exiting on its own — an OOM, a crash, the user killing it — cleared the
   * runtime's state and rebuilt the tray, but told the window nothing. The sidebar went on
   * naming a model that was gone, and the composer stayed enabled until someone tried to use it.
   */
  llama.on('status', (s: { phase?: string }) => {
    if (s?.phase === 'exited' && !llama.loaded) emitToSurfaces('model:status', null)
  })
  tray.on('double-click', () => {
    mainWindow?.show()
    mainWindow?.focus()
  })
}

/** Register every bridge handler with ipcMain, so the desktop and remote share one surface. */
function registerIpc(): void {
  for (const channel of Object.keys(handlers)) {
    ipcMain.handle(channel, async (_event, ...args: unknown[]) => invokeBridge(channel, args))
  }
  // Fire-and-forget variant for the permission reply, which must not block the renderer.
  ipcMain.on('agent:permission-response', (_e, id: string, decision: string) => {
    void invokeBridge('agent:permission-response', [id, decision])
  })
}

async function firstRunChecks(): Promise<void> {
  const proposal = await checkRelocation()
  if (!proposal) return

  const gb = (proposal.totalBytes / 1024 ** 3).toFixed(1)
  const choice = dialog.showMessageBoxSync({
    type: 'question',
    buttons: ['Move them here', 'Keep them there', 'Choose a folder'],
    defaultId: 0,
    title: 'Models folder found elsewhere',
    message: 'Your models are not beside the app.',
    /*
     * Three distinct paths, each labelled as what it actually is.
     *
     * This printed the destination *models folder* under "The app now lives at", so the dialog
     * claimed the application was inside a folder called LLMManagerModels — a folder which, in
     * the report that found this, did not exist at all. Being asked to approve moving seventeen
     * gigabytes on the strength of a path that is not what it says it is is not a decision
     * anyone can make correctly.
     */
    detail:
      `Found ${proposal.fileCount} files (${gb} GB) at:\n${proposal.from}\n\n` +
      `The app is now at:\n${proposal.appDir}\n\n` +
      `Moving them puts them at:\n${proposal.to}\n\n` +
      (proposal.sameVolume
        ? 'Both are on the same drive, so moving is instant.'
        : 'These are on different drives, so moving means a real copy and will take a while.')
  })

  if (choice === 1) {
    await invokeBridge('relocation:keep', [proposal.from])
    patchSettings({ modelsDir: proposal.from })
    return
  }

  if (choice === 2) {
    const picked = await dialog.showOpenDialog({ properties: ['openDirectory'] })
    if (!picked.canceled && picked.filePaths[0]) {
      patchSettings({ modelsDir: picked.filePaths[0] })
      await invokeBridge('relocation:keep', [picked.filePaths[0]])
    }
    return
  }

  await invokeBridge('relocation:move', [proposal])
}

app.whenReady().then(async () => {
  installCrashHandlers()
  ensureDirs()
  getDb()

  // Old staged uploads are cleared in the background; nothing waits on it. After ensureDirs so
  // the directory exists, and unawaited so a slow disk never delays the window.
  void handlers['attachments:prune']?.()

  setEmitter(emitToSurfaces)

  registerIpc()
  createWindow()
  createTray()

  downloadQueue.setToken(getHfToken())
  downloadQueue.setConnections(loadSettings().downloads.connections)
  downloadQueue.recoverOnStart()
  // Sweep .partial files with no queue entry behind them. recoverOnStart has just marked
  // everything resumable as paused, so whatever is left is unreachable garbage — a cancelled
  // or crashed download that nothing can ever resume, sitting on many GB of disk.
  void downloadQueue.cleanPartials(modelsDir()).then(
    ({ removed, bytes }) => {
      if (removed) logger.info('downloads', `removed ${removed} orphaned partial(s), ${bytes} bytes`)
    },
    (err) => logger.warn('downloads', 'could not clean orphaned partials', err)
  )
  /*
   * Progress goes to both surfaces.
   *
   * These five were sent straight to the desktop window, so a remote browser watching a download
   * saw a queue that never moved, a library that never refreshed after one finished, and MCP
   * status that stayed blank — while the bridge's own events reached it fine. Parity is only
   * true if everything uses the same emitter.
   */
  downloadQueue.on('update', (list) => emitToSurfaces('downloads:update', list))
  downloadQueue.on('completed', () => {
    void scanLibrary(modelsDir()).then((models) => {
      setLibrary(models)
      emitToSurfaces('library:update', models)
    })
  })

  await firstRunChecks()

  // Hardware detection runs on first launch without a wizard, so the library's compatibility
  // badges are accurate the moment it is opened.
  void detectHardware()
    .then((hw) => {
      logger.info('hardware', `backend=${hw.backend}`, hw.gpus.map((g) => g.name))
      // Log where the bundled binaries were looked for. A wrong vendor root is indistinguishable
      // from missing downloads in the UI, so record the resolved path on every launch.
      const vendor = vendorDiagnostics(hw.backend)
      logger.info('vendor', `root=${vendor.root} exists=${vendor.rootExists}`, {
        present: vendor.present,
        missing: vendor.missing
      })
      emitToSurfaces('hardware:update', hw)
      // From here the volatile figures — free VRAM, GPU load, free RAM — keep themselves current.
      startHardwareRefresh()
    })
    .catch((err) => logger.warn('hardware', 'detection failed', String(err)))

  void scanLibrary(modelsDir())
    .then((models) => {
      setLibrary(models)
      emitToSurfaces('library:update', models)
    })
    .catch((err) => logger.warn('library', 'scan failed', String(err)))

  void mcpManager.connectAll().then((results) => {
    if (results.length) logger.info('mcp', `connected ${results.filter((r) => r.ok).length}/${results.length}`)
    emitToSurfaces('mcp:update', mcpManager.status())
  })

  // Restart the API server if it was running when the app last closed.
  const settings = loadSettings()
  if (settings.server.enabled) {
    void invokeBridge('server:start', []).catch((err) => logger.warn('api', 'autostart failed', String(err)))
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
    else mainWindow?.show()
  })
})

app.on('window-all-closed', () => {
  // Deliberately does not quit: the tray keeps the model and API alive.
  if (process.platform === 'darwin') return
  if (loadSettings().ui.closeAction === 'quit') app.quit()
})

app.on('before-quit', async (event) => {
  if (shuttingDown) return
  shuttingDown = true
  userChoseQuit = true
  event.preventDefault()
  logger.info('app', 'shutting down')
  stopHardwareRefresh()
  await shutdown()
  app.exit(0)
})
