/**
 * Driving the user's own Firefox.
 *
 * Firefox speaks WebDriver BiDi over a WebSocket when it is started with
 * `--remote-debugging-port`, against its ordinary profile — the user's logins, extensions and open
 * tabs. Playwright cannot attach to a stock Firefox (it needs its own patched build), so this is a
 * small client for the handful of commands the browser tools need: list and open tabs, navigate,
 * run script, screenshot, and real mouse and keyboard input.
 *
 * Nothing here imports Electron, so it can be exercised against a throwaway profile from a script.
 */

export interface BidiTab {
  id: string
  url: string
}

interface Pending {
  resolve: (value: unknown) => void
  reject: (err: Error) => void
  timer: ReturnType<typeof setTimeout>
}

/** WebDriver's names for the keys that are not characters. */
const KEYS: Record<string, string> = {
  Enter: '',
  Tab: '',
  Backspace: '',
  Delete: '',
  Escape: '',
  Control: '',
  ArrowUp: '',
  ArrowDown: '',
  ArrowLeft: '',
  ArrowRight: '',
  PageDown: '',
  PageUp: '',
  Home: '',
  End: ''
}

export class BidiSession {
  private socket: WebSocket | null = null
  private nextId = 1
  private pending = new Map<number, Pending>()
  private closed = false

  /** Connect and open a session, or throw if nothing is listening there. */
  static async connect(port: number, timeoutMs = 4000): Promise<BidiSession> {
    const session = new BidiSession()
    await session.open(`ws://127.0.0.1:${port}/session`, timeoutMs)
    await session.send('session.new', { capabilities: {} })
    return session
  }

  private open(url: string, timeoutMs: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(url)
      const timer = setTimeout(() => {
        socket.close()
        reject(new Error('timed out'))
      }, timeoutMs)
      socket.addEventListener('open', () => {
        clearTimeout(timer)
        this.socket = socket
        resolve()
      })
      socket.addEventListener('error', () => {
        clearTimeout(timer)
        reject(new Error('could not connect'))
      })
      socket.addEventListener('close', () => {
        this.closed = true
        for (const p of this.pending.values()) {
          clearTimeout(p.timer)
          p.reject(new Error('The browser closed the connection.'))
        }
        this.pending.clear()
      })
      socket.addEventListener('message', (ev) => {
        let msg: { id?: number; type?: string; result?: unknown; error?: string; message?: string }
        try {
          msg = JSON.parse(String(ev.data))
        } catch {
          return
        }
        if (typeof msg.id !== 'number') return // an event; none are subscribed to
        const p = this.pending.get(msg.id)
        if (!p) return
        this.pending.delete(msg.id)
        clearTimeout(p.timer)
        if (msg.type === 'error' || msg.error) p.reject(new Error(msg.message || msg.error || 'the browser refused'))
        else p.resolve(msg.result)
      })
    })
  }

  get connected(): boolean {
    return !!this.socket && !this.closed
  }

  send<T = Record<string, unknown>>(method: string, params: Record<string, unknown>, timeoutMs = 30_000): Promise<T> {
    if (!this.socket || this.closed) return Promise.reject(new Error('The browser is no longer connected.'))
    const id = this.nextId++
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`The browser did not answer ${method} in time.`))
      }, timeoutMs)
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, timer })
      this.socket!.send(JSON.stringify({ id, method, params }))
    })
  }

  /** End the session and hang up. The browser itself stays open. */
  async close(): Promise<void> {
    if (!this.socket || this.closed) return
    await this.send('session.end', {}, 3000).catch(() => undefined)
    this.socket.close()
    this.closed = true
  }

  // ---------------------------------------------------------------- tabs

  async tabs(): Promise<BidiTab[]> {
    const tree = await this.send<{ contexts: { context: string; url: string }[] }>('browsingContext.getTree', { maxDepth: 0 })
    return tree.contexts.map((c) => ({ id: c.context, url: c.url }))
  }

  async newTab(): Promise<string> {
    const made = await this.send<{ context: string }>('browsingContext.create', { type: 'tab' })
    return made.context
  }

  async activate(tab: string): Promise<void> {
    await this.send('browsingContext.activate', { context: tab })
  }

  async closeTab(tab: string): Promise<void> {
    await this.send('browsingContext.close', { context: tab })
  }

  async navigate(tab: string, url: string, timeoutMs: number): Promise<void> {
    await this.send('browsingContext.navigate', { context: tab, url, wait: 'interactive' }, timeoutMs)
  }

  // ---------------------------------------------------------------- script

  /**
   * Run script in a tab and return its value.
   *
   * The value comes back as JSON text rather than through BiDi's own serialisation, which wraps
   * every object in type tags. A script that evaluates to a function is called, as Playwright does,
   * so the same strings work against either browser.
   */
  async evaluate<T = unknown>(tab: string, script: string, timeoutMs = 30_000): Promise<T> {
    const expression =
      `(async () => { let __v = (${script}\n); if (typeof __v === 'function') __v = __v(); ` +
      `__v = await __v; return JSON.stringify(__v === undefined ? null : __v) })()`
    const r = await this.send<{
      type: string
      result?: { type: string; value?: string }
      exceptionDetails?: { text?: string }
    }>('script.evaluate', { expression, target: { context: tab }, awaitPromise: true, resultOwnership: 'none' }, timeoutMs)
    if (r.type !== 'success') throw new Error(r.exceptionDetails?.text || 'The script threw an error.')
    const text = r.result?.value
    return (typeof text === 'string' ? JSON.parse(text) : null) as T
  }

  // ---------------------------------------------------------------- input

  /** A real click at a point in the tab's viewport. */
  async clickAt(tab: string, x: number, y: number): Promise<void> {
    await this.send('input.performActions', {
      context: tab,
      actions: [
        {
          type: 'pointer',
          id: 'mouse',
          parameters: { pointerType: 'mouse' },
          actions: [
            { type: 'pointerMove', x: Math.round(x), y: Math.round(y), origin: 'viewport' },
            { type: 'pointerDown', button: 0 },
            { type: 'pointerUp', button: 0 }
          ]
        }
      ]
    })
    await this.send('input.releaseActions', { context: tab }).catch(() => undefined)
  }

  /** Press keys, each either a single character or a name such as Enter. */
  async press(tab: string, keys: string[]): Promise<void> {
    const actions: { type: string; value: string }[] = []
    for (const key of keys) {
      const value = KEYS[key] ?? key
      actions.push({ type: 'keyDown', value }, { type: 'keyUp', value })
    }
    await this.send('input.performActions', { context: tab, actions: [{ type: 'key', id: 'keyboard', actions }] })
    await this.send('input.releaseActions', { context: tab }).catch(() => undefined)
  }

  /** Type text as real key presses, a line break becoming Enter. */
  async type(tab: string, text: string): Promise<void> {
    const keys = [...text.replace(/\r\n/g, '\n')].map((ch) => (ch === '\n' ? 'Enter' : ch))
    // In pieces, so one very long string is not one very long message.
    for (let i = 0; i < keys.length; i += 200) await this.press(tab, keys.slice(i, i + 200))
  }

  async screenshot(tab: string, fullPage: boolean): Promise<Buffer> {
    const shot = await this.send<{ data: string }>(
      'browsingContext.captureScreenshot',
      { context: tab, origin: fullPage ? 'document' : 'viewport' },
      60_000
    )
    return Buffer.from(shot.data, 'base64')
  }
}
