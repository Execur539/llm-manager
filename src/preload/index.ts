/**
 * Preload: the only bridge between renderer and main.
 * contextIsolation is on and nodeIntegration is off, so the renderer gets exactly this
 * surface and nothing else.
 */

import { contextBridge, ipcRenderer } from 'electron'
import type { AppSettings, PermissionDecision } from '@shared/types'

const api = {
  settings: {
    get: () => ipcRenderer.invoke('settings:get'),
    patch: (patch: Partial<AppSettings>) => ipcRenderer.invoke('settings:patch', patch)
  },
  hardware: {
    get: (refresh?: boolean) => ipcRenderer.invoke('hardware:get', refresh),
    onUpdate: (cb: (hw: unknown) => void) => {
      ipcRenderer.on('hardware:update', (_e, hw) => cb(hw))
    }
  },
  runtime: {
    missingBinaries: () => ipcRenderer.invoke('runtime:missing-binaries')
  },
  library: {
    scan: () => ipcRenderer.invoke('library:scan'),
    disk: () => ipcRenderer.invoke('library:disk')
  },
  autofit: {
    plan: (modelId: string) => ipcRenderer.invoke('autofit:plan', modelId)
  },
  model: {
    load: (modelId: string, plan: unknown) => ipcRenderer.invoke('model:load', modelId, JSON.stringify(plan)),
    unload: () => ipcRenderer.invoke('model:unload'),
    status: () => ipcRenderer.invoke('model:status')
  },
  relocation: {
    check: () => ipcRenderer.invoke('relocation:check'),
    keep: (from: string) => ipcRenderer.invoke('relocation:keep', from),
    move: (proposal: unknown) => ipcRenderer.invoke('relocation:move', JSON.stringify(proposal)),
    cancel: () => ipcRenderer.send('relocation:cancel'),
    onProgress: (cb: (p: unknown) => void) => {
      ipcRenderer.on('relocation:progress', (_e, p) => cb(p))
    }
  },
  agent: {
    tools: () => ipcRenderer.invoke('agent:tools'),
    grammar: () => ipcRenderer.invoke('agent:grammar'),
    run: (session: unknown, input: string) => ipcRenderer.invoke('agent:run', JSON.stringify(session), input),
    stop: () => ipcRenderer.invoke('agent:stop'),
    setCwd: () => ipcRenderer.invoke('agent:set-cwd'),
    checkpoints: (sessionId: string) => ipcRenderer.invoke('agent:checkpoints', sessionId),
    rewind: (sessionId: string, checkpointId: string) => ipcRenderer.invoke('agent:rewind', sessionId, checkpointId),
    respondPermission: (id: string, decision: PermissionDecision) =>
      ipcRenderer.send('agent:permission-response', id, decision),
    onDelta: (cb: (text: string) => void) => ipcRenderer.on('agent:delta', (_e, t) => cb(t)),
    onMessage: (cb: (m: unknown) => void) => ipcRenderer.on('agent:message', (_e, m) => cb(m)),
    onToolCall: (cb: (c: unknown) => void) => ipcRenderer.on('agent:tool-call', (_e, c) => cb(c)),
    onToolResult: (cb: (r: unknown) => void) => ipcRenderer.on('agent:tool-result', (_e, r) => cb(r)),
    onPermissionRequest: (cb: (r: unknown) => void) => ipcRenderer.on('agent:permission-request', (_e, r) => cb(r)),
    onDone: (cb: (reason: string) => void) => ipcRenderer.on('agent:done', (_e, r) => cb(r)),
    onError: (cb: (err: string) => void) => ipcRenderer.on('agent:error', (_e, err) => cb(err))
  }
}

contextBridge.exposeInMainWorld('api', api)

export type Api = typeof api
