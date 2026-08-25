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
import { handlers, invokeBridge, setEmitter, setLibrary, shutdown, modelsDir } from './bridge'
import { installCrashHandlers, logger } from './log'

let mainWindow: BrowserWindow | null = null
let tray: Tray | null = null
let quitting = false

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1040,
    minHeight: 660,
    show: false,
    backgroundColor: '#0f1115',
    title: 'LLM Manager',
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  })

  mainWindow.on('ready-to-show', () => mainWindow?.show())

  // Round 11: ask on first close, then remember the answer.
  mainWindow.on('close', (event) => {
    if (quitting) return
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
      quitting = true
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
  // A 1x1 transparent image is a valid tray icon; a real one replaces this at branding time.
  const icon = nativeImage.createFromDataURL(
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAQUlEQVR42mNkYPhfz0AEYBxVSF+FjIyM/xkYGP4TowGmiGgFyIqIVoBLEVEKcCkiSgE+RQQV4FOEVwEhRXgVAABtaB1V8bkLbwAAAABJRU5ErkJggg=='
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
          click: () => void llama.unload()
        },
        { type: 'separator' },
        {
          label: 'Quit',
          click: () => {
            quitting = true
            app.quit()
          }
        }
      ])
    )
  }

  rebuild()
  llama.on('status', rebuild)
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
    detail:
      `Found ${proposal.fileCount} files (${gb} GB) at:\n${proposal.from}\n\n` +
      `The app now lives at:\n${proposal.to}\n\n` +
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

  // One emitter feeds both surfaces: the desktop window over IPC, and every connected
  // remote browser over SSE. Neither can silently miss an event the other gets.
  setEmitter((channel, payload) => {
    if (!mainWindow?.isDestroyed()) mainWindow?.webContents.send(channel, payload)
    remoteWeb.broadcast(channel, payload)
  })

  registerIpc()
  createWindow()
  createTray()

  downloadQueue.setToken(getHfToken())
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
  downloadQueue.on('update', (list) => mainWindow?.webContents.send('downloads:update', list))
  downloadQueue.on('completed', () => {
    void scanLibrary(modelsDir()).then((models) => {
      setLibrary(models)
      mainWindow?.webContents.send('library:update', models)
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
      mainWindow?.webContents.send('hardware:update', hw)
    })
    .catch((err) => logger.warn('hardware', 'detection failed', String(err)))

  void scanLibrary(modelsDir())
    .then((models) => {
      setLibrary(models)
      mainWindow?.webContents.send('library:update', models)
    })
    .catch((err) => logger.warn('library', 'scan failed', String(err)))

  void mcpManager.connectAll().then((results) => {
    if (results.length) logger.info('mcp', `connected ${results.filter((r) => r.ok).length}/${results.length}`)
    mainWindow?.webContents.send('mcp:update', mcpManager.status())
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
  if (quitting) return
  quitting = true
  event.preventDefault()
  logger.info('app', 'shutting down')
  await shutdown()
  app.exit(0)
})
