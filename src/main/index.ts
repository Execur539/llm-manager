/**
 * Main process entry point.
 */

import { app, BrowserWindow, ipcMain, dialog } from 'electron'
import path from 'node:path'
import type { AppSettings, FitResult, ModelRecord, PermissionDecision, PermissionRequest } from '@shared/types'
import { ensureDirs, defaultModelsDir } from './storage/paths'
import { loadSettings, patchSettings } from './storage/settings'
import { detectHardware, refreshFreeVram } from './hardware/gpu'
import { scanLibrary, libraryDiskUsage } from './models/library'
import { checkRelocation, keepInPlace, performMove } from './models/relocation'
import { planFit, DEFAULT_CONSTRAINTS, fmtBytes } from './autofit/engine'
import { llama } from './runtime/llama'
import { missingBinaries } from './runtime/binaries'
import { Agent } from './agent/loop'
import { killAllJobs } from './agent/tools/exec'
import { listCheckpoints, rewindTo } from './agent/checkpoints'
import type { HardwareSnapshot } from '@shared/types'

let mainWindow: BrowserWindow | null = null
let hardware: HardwareSnapshot | null = null
let library: ModelRecord[] = []
let agent: Agent | null = null

/** Pending permission prompts, keyed by request id, resolved when the renderer answers. */
const pendingPermissions = new Map<string, (d: PermissionDecision) => void>()

function modelsDir(): string {
  return loadSettings().modelsDir ?? defaultModelsDir()
}

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 640,
    show: false,
    backgroundColor: '#0f1115',
    title: 'LLM Manager',
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  })

  mainWindow.on('ready-to-show', () => mainWindow?.show())

  if (process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'))
  }
}

function send(channel: string, payload: unknown): void {
  mainWindow?.webContents.send(channel, payload)
}

function getAgent(): Agent {
  const s = loadSettings()
  if (!agent) {
    agent = new Agent({
      cwd: process.cwd(),
      planMode: s.agent.planMode,
      maxToolCallsPerTurn: s.agent.maxToolCallsPerTurn,
      commandTimeoutMs: s.agent.commandTimeoutMs,
      hardBlocksDisabled: s.agent.hardBlocksDisabled,
      hfToken: s.hfToken,
      requestPermission: (req: PermissionRequest) =>
        new Promise<PermissionDecision>((resolve) => {
          pendingPermissions.set(req.id, resolve)
          send('agent:permission-request', req)
        })
    })

    agent.on('delta', (text: string) => send('agent:delta', text))
    agent.on('message', (m) => send('agent:message', m))
    agent.on('toolCall', (c) => send('agent:tool-call', c))
    agent.on('toolResult', (r) => send('agent:tool-result', r))
    agent.on('done', (reason) => send('agent:done', reason))
    agent.on('error', (e) => send('agent:error', e))
  }
  return agent
}

function registerIpc(): void {
  ipcMain.handle('settings:get', () => loadSettings())
  ipcMain.handle('settings:patch', (_e, patch: Partial<AppSettings>) => patchSettings(patch))

  ipcMain.handle('hardware:get', async (_e, refresh?: boolean) => {
    if (!hardware || refresh) hardware = await detectHardware()
    else hardware = await refreshFreeVram(hardware)
    return hardware
  })

  ipcMain.handle('runtime:missing-binaries', async () => {
    if (!hardware) hardware = await detectHardware()
    return missingBinaries(hardware.backend)
  })

  ipcMain.handle('library:scan', async () => {
    library = await scanLibrary(modelsDir())
    return library
  })

  ipcMain.handle('library:disk', async () => libraryDiskUsage(modelsDir()))

  /**
   * Compute a fit for one model. Free VRAM is re-measured immediately before planning —
   * this is the P1 fix, and a snapshot from app start would defeat it.
   */
  ipcMain.handle('autofit:plan', async (_e, modelId: string): Promise<FitResult | { error: string }> => {
    const model = library.find((m) => m.id === modelId)
    if (!model) return { error: 'Model not found' }
    if (!model.arch) return { error: model.error ?? 'Model metadata could not be parsed' }

    if (!hardware) hardware = await detectHardware()
    hardware = await refreshFreeVram(hardware)

    const s = loadSettings()
    return planFit(model.arch, hardware, {
      ...DEFAULT_CONSTRAINTS,
      minKvType: s.autoFit.minKvType,
      preferredKvType: s.autoFit.preferredKvType,
      targetContext: s.autoFit.targetContext,
      idealContext: s.autoFit.idealContext,
      headroomBytes: s.autoFit.headroomMb * 1024 * 1024,
      overrides: {}
    })
  })

  ipcMain.handle('model:load', async (_e, modelId: string, planJson: string) => {
    const model = library.find((m) => m.id === modelId)
    if (!model) throw new Error('Model not found')
    if (!hardware) hardware = await detectHardware()
    const plan = JSON.parse(planJson)
    const loaded = await llama.load(model, plan, hardware.backend)
    return { port: loaded.port, model: loaded.model.filename, context: plan.contextLength }
  })

  ipcMain.handle('model:unload', async () => {
    await llama.unload()
    return true
  })

  ipcMain.handle('model:status', () => {
    const l = llama.loaded
    return l ? { model: l.model.filename, port: l.port, plan: l.plan, startedAt: l.startedAt } : null
  })

  ipcMain.handle('relocation:check', () => checkRelocation())
  ipcMain.handle('relocation:keep', (_e, from: string) => {
    keepInPlace(from)
    return true
  })
  ipcMain.handle('relocation:move', async (_e, proposalJson: string) => {
    const proposal = JSON.parse(proposalJson)
    let cancelled = false
    ipcMain.once('relocation:cancel', () => {
      cancelled = true
    })
    return performMove(proposal, (p) => send('relocation:progress', p), () => cancelled)
  })

  // ------------------------------------------------------------------ agent
  ipcMain.handle('agent:tools', () => getAgent().listTools())
  ipcMain.handle('agent:grammar', () => getAgent().grammar())

  ipcMain.handle('agent:run', async (_e, sessionJson: string, input: string) => {
    const session = JSON.parse(sessionJson)
    const s = loadSettings()
    const a = getAgent()
    a.updateOptions({
      planMode: s.agent.planMode,
      maxToolCallsPerTurn: s.agent.maxToolCallsPerTurn,
      commandTimeoutMs: s.agent.commandTimeoutMs,
      hardBlocksDisabled: s.agent.hardBlocksDisabled,
      hfToken: s.hfToken
    })
    await a.run(session, input)
    return session
  })

  ipcMain.handle('agent:stop', () => {
    agent?.stop()
    return true
  })

  ipcMain.on('agent:permission-response', (_e, id: string, decision: PermissionDecision) => {
    const resolve = pendingPermissions.get(id)
    if (resolve) {
      pendingPermissions.delete(id)
      resolve(decision)
    }
  })

  ipcMain.handle('agent:set-cwd', async () => {
    const result = await dialog.showOpenDialog({ properties: ['openDirectory'] })
    if (result.canceled || !result.filePaths[0]) return null
    getAgent().updateOptions({ cwd: result.filePaths[0] })
    return result.filePaths[0]
  })

  ipcMain.handle('agent:checkpoints', (_e, sessionId: string) => listCheckpoints(sessionId))
  ipcMain.handle('agent:rewind', (_e, sessionId: string, checkpointId: string) => rewindTo(sessionId, checkpointId))

  ipcMain.handle('util:fmt-bytes', (_e, n: number) => fmtBytes(n))
}

app.whenReady().then(async () => {
  ensureDirs()
  registerIpc()
  createWindow()

  // Warm the hardware snapshot so compatibility badges are accurate the moment the
  // library opens — the plan calls for no wizard, but detection still runs on first launch.
  detectHardware()
    .then((hw) => {
      hardware = hw
      send('hardware:update', hw)
    })
    .catch(() => undefined)

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', async () => {
  killAllJobs()
  await llama.unload()
})
