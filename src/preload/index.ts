/**
 * Preload: the only bridge between renderer and main.
 *
 * Deliberately generic — `invoke(channel, ...args)` mirrors the bridge handler map rather than
 * enumerating a second API surface that could drift from it. The remote web UI implements the
 * same shape over HTTP, so the React app cannot tell which one it is running against.
 */

import { contextBridge, ipcRenderer, webUtils } from 'electron'

/** Channels the renderer is allowed to subscribe to. */
const EVENT_CHANNELS = [
  'hardware:update',
  'library:update',
  'model:status',
  'autofit:verified',
  'relocation:progress',
  'downloads:update',
  'chat:delta',
  'chat:reasoning',
  'chat:prompt-progress',
  'chat:context',
  'chat:message',
  'chat:notes',
  'agent:delta',
  'agent:reasoning',
  'agent:prompt-progress',
  'agent:context',
  'agent:message',
  'agent:tool-call',
  'agent:tool-call-partial',
  'agent:tool-result',
  'agent:sub-tool-call',
  'agent:permission-request',
  // The agent pausing to ask the user something, as distinct from asking permission.
  'agent:question',
  'agent:compacting',
  'agent:compacted',
  'agent:done',
  'agent:error',
  // Ultra's attempts, for both surfaces. The renderer subscribes to every channel in this list
  // at import time and throws on one it does not recognise, so a new event has to be added here
  // before it can be listened for — the whole renderer fails to mount otherwise.
  'chat:ultra-sample-start',
  'chat:ultra-sample-delta',
  'chat:ultra-sample-reasoning',
  'chat:ultra-sample',
  'chat:ultra-synthesis',
  'agent:ultra-sample-start',
  'agent:ultra-sample-delta',
  'agent:ultra-sample-reasoning',
  'agent:ultra-sample',
  'agent:ultra-synthesis',
  'agent:ultra-plan',
  'mcp:update',
  'rag:progress',
  'server:status',
  'remote:status',
  'update:progress'
] as const

export type EventChannel = (typeof EVENT_CHANNELS)[number]

const api = {
  /** Call any bridge handler. Mirrors the main-process handler map exactly. */
  invoke: <T = unknown>(channel: string, ...args: unknown[]): Promise<T> =>
    ipcRenderer.invoke(channel, ...args) as Promise<T>,

  /** Fire-and-forget, used for the permission reply so the renderer never blocks. */
  send: (channel: string, ...args: unknown[]): void => {
    ipcRenderer.send(channel, ...args)
  },

  /** Subscribe to a main-process event. Returns an unsubscribe function. */
  on: (channel: EventChannel, cb: (payload: never) => void): (() => void) => {
    if (!EVENT_CHANNELS.includes(channel)) {
      throw new Error(`Refusing to subscribe to unknown channel: ${channel}`)
    }
    const listener = (_e: Electron.IpcRendererEvent, payload: unknown): void => cb(payload as never)
    ipcRenderer.on(channel, listener)
    return () => ipcRenderer.removeListener(channel, listener)
  },

  /**
   * The on-disk path of a dropped File.
   *
   * `File.path` was removed in Electron 32, and it is the only way to attach a 4 GB video
   * without reading it into memory first. Returns '' for anything the browser synthesised
   * (a paste, a remote session), which is the caller's cue to send the bytes instead.
   */
  pathForFile: (file: File): string => {
    try {
      return webUtils.getPathForFile(file)
    } catch {
      return ''
    }
  },

  /** Tells the renderer it is running in the desktop shell rather than a remote browser. */
  isDesktop: true
}

contextBridge.exposeInMainWorld('api', api)

export type Api = typeof api
