/**
 * Client-side bridge.
 *
 * In the desktop shell this forwards to the preload's IPC bridge. In a remote browser there is
 * no preload, so it POSTs to /bridge and polls /events instead. The React app above this line
 * cannot tell the difference, which is what keeps the two UIs from drifting.
 */

export type EventChannel = string

interface DesktopApi {
  invoke: <T>(channel: string, ...args: unknown[]) => Promise<T>
  send: (channel: string, ...args: unknown[]) => void
  on: (channel: EventChannel, cb: (payload: never) => void) => () => void
  isDesktop: boolean
}

declare global {
  interface Window {
    api?: DesktopApi
  }
}

const listeners = new Map<string, Set<(payload: unknown) => void>>()

/** Remote transport: bridge calls over HTTP, events over Server-Sent Events. */
class RemoteTransport {
  private source: EventSource | null = null

  async invoke<T>(channel: string, ...args: unknown[]): Promise<T> {
    const res = await fetch('/bridge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ channel, args })
    })
    if (res.status === 401) {
      window.location.reload() // session expired; the server will show the login page
      throw new Error('Session expired')
    }
    const json = (await res.json()) as { result?: T; error?: string }
    if (json.error) throw new Error(json.error)
    return json.result as T
  }

  connect(): void {
    if (this.source) return
    try {
      this.source = new EventSource('/events')
      this.source.onmessage = (ev) => {
        try {
          const { channel, payload } = JSON.parse(ev.data) as { channel: string; payload: unknown }
          listeners.get(channel)?.forEach((cb) => cb(payload))
        } catch {
          /* malformed frame */
        }
      }
      this.source.onerror = () => {
        // EventSource reconnects on its own; nothing to do but let it.
      }
    } catch {
      this.source = null
    }
  }
}

const remote = new RemoteTransport()

export const isDesktop = typeof window !== 'undefined' && !!window.api?.isDesktop

export async function invoke<T = unknown>(channel: string, ...args: unknown[]): Promise<T> {
  if (window.api) return window.api.invoke<T>(channel, ...args)
  return remote.invoke<T>(channel, ...args)
}

export function send(channel: string, ...args: unknown[]): void {
  if (window.api) {
    window.api.send(channel, ...args)
    return
  }
  void remote.invoke(channel, ...args)
}

export function on<T = unknown>(channel: EventChannel, cb: (payload: T) => void): () => void {
  if (window.api) return window.api.on(channel, cb as (p: never) => void)

  remote.connect()
  const set = listeners.get(channel) ?? new Set()
  set.add(cb as (p: unknown) => void)
  listeners.set(channel, set)
  return () => set.delete(cb as (p: unknown) => void)
}

// ---------------------------------------------------------------- formatting helpers

export function fmtBytes(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '—'
  if (n < 0) return 'unknown'
  const gb = n / 1024 ** 3
  if (gb >= 1) return `${gb.toFixed(gb >= 10 ? 1 : 2)} GB`
  const mb = n / 1024 ** 2
  if (mb >= 1) return `${mb.toFixed(0)} MB`
  return `${n} B`
}

export function fmtTokens(n: number): string {
  if (n >= 1024) return `${(n / 1024).toFixed(n >= 10240 ? 0 : 1)}K`
  return String(n)
}

export function fmtDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(1)}s`
  const m = Math.floor(s / 60)
  return `${m}m ${Math.round(s % 60)}s`
}

export function fmtRelative(ts: number): string {
  const diff = Date.now() - ts
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days < 30) return `${days}d ago`
  return new Date(ts).toLocaleDateString()
}
