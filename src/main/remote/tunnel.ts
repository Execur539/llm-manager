/**
 * Cloudflare tunnel supervision.
 *
 * Zero-config remote access: `cloudflared tunnel --url http://127.0.0.1:<port>` returns a
 * public HTTPS hostname with TLS terminated by Cloudflare, needs no account, and works behind
 * CGNAT. The URL is parsed out of cloudflared's own stderr, which is where it prints it.
 */

import { spawn, ChildProcess } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { runtimeBinary } from '../runtime/binaries'

export interface TunnelState {
  running: boolean
  url: string | null
  error: string | null
}

const URL_RE = /https:\/\/[a-z0-9-]+\.trycloudflare\.com/i

class TunnelManager extends EventEmitter {
  private child: ChildProcess | null = null
  private state: TunnelState = { running: false, url: null, error: null }

  get current(): TunnelState {
    return this.state
  }

  private update(patch: Partial<TunnelState>): void {
    this.state = { ...this.state, ...patch }
    this.emit('state', this.state)
  }

  /** Start a quick tunnel to the given local port and resolve once the URL appears. */
  async start(localPort: number): Promise<TunnelState> {
    await this.stop()

    const exe = runtimeBinary('cloudflared')
    const child = spawn(
      exe,
      ['tunnel', '--no-autoupdate', '--url', `http://127.0.0.1:${localPort}`],
      { windowsHide: true }
    )
    this.child = child
    this.update({ running: true, url: null, error: null })

    return new Promise<TunnelState>((resolve, reject) => {
      let settled = false
      const timer = setTimeout(() => {
        if (settled) return
        settled = true
        this.update({ error: 'Tunnel did not produce a URL within 60 seconds' })
        reject(new Error(this.state.error as string))
      }, 60000)

      const onData = (buf: Buffer): void => {
        const text = buf.toString()
        this.emit('log', text)
        const match = text.match(URL_RE)
        if (match && !settled) {
          settled = true
          clearTimeout(timer)
          this.update({ url: match[0], running: true, error: null })
          resolve(this.state)
        }
      }

      child.stdout?.on('data', onData)
      child.stderr?.on('data', onData)

      child.on('error', (err) => {
        if (settled) return
        settled = true
        clearTimeout(timer)
        const message = err.message.includes('ENOENT')
          ? 'cloudflared is not bundled yet. Run `npm run fetch-vendor`.'
          : err.message
        this.update({ running: false, error: message })
        reject(new Error(message))
      })

      child.on('exit', (code) => {
        this.child = null
        this.update({ running: false, url: null, error: code ? `cloudflared exited (${code})` : null })
        if (!settled) {
          settled = true
          clearTimeout(timer)
          reject(new Error(this.state.error ?? 'cloudflared exited before producing a URL'))
        }
      })
    })
  }

  async stop(): Promise<void> {
    const child = this.child
    this.child = null
    this.update({ running: false, url: null })
    if (!child) return
    await new Promise<void>((resolve) => {
      child.once('exit', () => resolve())
      child.kill()
      setTimeout(resolve, 3000)
    })
  }
}

export const tunnel = new TunnelManager()
