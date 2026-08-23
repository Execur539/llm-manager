/**
 * Preload: the only bridge between renderer and main.
 *
 * Deliberately generic — `invoke(channel, ...args)` mirrors the bridge handler map rather than
 * enumerating a second API surface that could drift from it. The remote web UI implements the
 * same shape over HTTP, so the React app cannot tell which one it is running against.
 */

import { contextBridge, ipcRenderer } from 'electron'

/** Channels the renderer is allowed to subscribe to. */
const EVENT_CHANNELS = [
  'hardware:update',
  'library:update',
  'model:status',
  'autofit:verified',
  'relocation:progress',
  'downloads:update',
  'chat:delta',
  'chat:message',
  'chat:notes',
  'agent:delta',
  'agent:message',
  'agent:tool-call',
  'agent:tool-result',
  'agent:sub-tool-call',
  'agent:permission-request',
  'agent:compacted',
  'agent:done',
  'agent:error',
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

  /** Tells the renderer it is running in the desktop shell rather than a remote browser. */
  isDesktop: true
}

contextBridge.exposeInMainWorld('api', api)

export type Api = typeof api
