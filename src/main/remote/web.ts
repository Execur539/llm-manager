/**
 * The remote web UI server.
 *
 * Serves the same React bundle the desktop window uses, behind password + session auth, and
 * proxies a JSON-RPC-ish bridge onto the same handlers the desktop IPC uses. That is what
 * makes "full parity" real rather than a second, thinner app.
 *
 * Destructive actions are gated: they are refused here and must be confirmed on the desktop.
 * The gate lives server-side so a modified client cannot talk its way past it.
 */

import http from 'node:http'
import https from 'node:https'
import fs from 'node:fs'
import path from 'node:path'
import { app } from 'electron'
import { loadSettings } from '../storage/settings'
import { resolveMedia, mediaHeaders, mediaStream } from '../chat/media'
import {
  isLockedOut,
  issueSession,
  recordFailure,
  recordSuccess,
  hasPassword,
  verifyPassword,
  verifySession
} from './auth'

/** Actions that require physical access to the desktop app. */
const DESKTOP_ONLY = new Set([
  'library:delete-model',
  'settings:set-password',
  'remote:disable',
  'remote:enable',
  'mcp:add',
  'mcp:remove',
  // Replaces the executable and relaunches it. The URL is verified against the release feed
  // regardless, but deciding to replace the app is not something to do from somewhere else.
  'update:apply',
  // Quitting the host from a remote tab is not a feature.
  'app:quit',
  /*
   * These open a native dialog on the desktop and await it.
   *
   * A remote caller cannot see or dismiss one, so the request hangs until somebody walks over to
   * the machine — and the dialog appears unbidden on someone else's screen either way.
   *
   * The list was originally written from the two obvious cases and then not revisited as more
   * pickers appeared, so `attachments:pick` and `rag:ingest` — both plain file dialogs — were
   * reachable remotely and would hang the request for exactly the documented reason.
   */
  'library:import',
  'chat:export',
  'agent:set-cwd',
  'attachments:pick',
  'rag:ingest',
  /*
   * These pop an Explorer window on the host.
   *
   * Nobody sitting in front of the machine asked for it, and the remote caller cannot see the
   * result, so the only effect a remote invocation has is on someone else's screen.
   */
  'diagnostics:reveal',
  'shell:reveal'
])

/*
 * Settings a remote session may not change, whatever route it takes.
 *
 * The deny-list above names channels, which was not enough. `settings:disable-hard-blocks` and
 * `agent:set-hard-blocks` were listed but have never existed as handlers, while the path the UI
 * actually uses — `settings:patch`, taking an arbitrary Partial<AppSettings> — was not listed at
 * all. A remote session could therefore turn off the hard-block list and grant itself write and
 * execute tools with two ordinary settings calls, which is precisely what both of those gates
 * were written to prevent.
 *
 * Checked by content rather than by channel, so gating stays here in the web server and the
 * handler map stays transport-agnostic.
 */
const PRIVILEGED_SETTINGS: Record<string, string[]> = {
  agent: ['hardBlocksDisabled', 'remoteToolsEnabled']
}

/*
 * Blocks a privileged *change*, not the mere presence of a privileged key.
 *
 * The settings UI patches by spreading a whole section — `{ agent: { ...agent, planMode } }` —
 * so every one of these keys travels with every agent patch, carrying its current value.
 * Rejecting on presence would have blocked a remote session from touching any agent setting at
 * all. Only a value that differs from what is already stored is an attempt to change it.
 */
function privilegedSettingsIn(patch: unknown): string | null {
  if (!patch || typeof patch !== 'object') return null
  const current = loadSettings() as unknown as Record<string, Record<string, unknown>>

  for (const [section, keys] of Object.entries(PRIVILEGED_SETTINGS)) {
    const value = (patch as Record<string, unknown>)[section]
    if (!value || typeof value !== 'object') continue
    for (const key of keys) {
      const incoming = (value as Record<string, unknown>)[key]
      if (!(key in (value as Record<string, unknown>))) continue
      if (incoming !== current?.[section]?.[key]) return `${section}.${key}`
    }
  }
  return null
}

export interface WebServerOptions {
  port: number
  /** invoke a bridge handler by name; the same map the desktop IPC uses */
  invoke: (channel: string, args: unknown[]) => Promise<unknown>
  /** TLS material for the own-domain path; omitted means plain HTTP behind a tunnel */
  tls?: { certPath: string; keyPath: string }
  /**
   * How this server is reached, which decides two things that must agree.
   *
   * Behind a tunnel, cloudflared connects from this machine, so the socket address is always
   * loopback and the real client is only knowable from `CF-Connecting-IP` — which Cloudflare
   * overwrites, so it can be trusted. Serving a domain directly, the socket address *is* the
   * client and that header is whatever the client typed.
   *
   * Getting this wrong is not cosmetic: the login lockout is keyed on the address. Trusting the
   * header on a directly-reachable server let anyone send a different value on every attempt and
   * guess the password without limit.
   */
  mode: 'tunnel' | 'own-domain'
}

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2'
}

function rendererRoot(): string {
  // Same bundle as the desktop window loads.
  return app.isPackaged
    ? path.join(process.resourcesPath, 'app.asar', 'out', 'renderer')
    : path.join(app.getAppPath(), 'out', 'renderer')
}

function parseCookies(header: string | undefined): Record<string, string> {
  const out: Record<string, string> = {}
  for (const part of (header ?? '').split(';')) {
    const [k, ...rest] = part.trim().split('=')
    if (k) out[k] = decodeURIComponent(rest.join('='))
  }
  return out
}

/*
 * Read a request body, refusing one that is too large.
 *
 * Rejecting is not enough on its own: the promise settles but the `data` listener keeps firing
 * and keeps concatenating, so an oversized upload carried on growing the string long after the
 * caller had given up on it. The socket is destroyed and the listener detached, which is what
 * makes the limit an actual limit.
 */
function readBody(req: http.IncomingMessage, limit = 32 * 1024 * 1024): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = ''
    const onData = (c: Buffer): void => {
      data += c
      if (data.length > limit) {
        req.off('data', onData)
        data = ''
        req.destroy()
        reject(new Error('body too large'))
      }
    }
    req.on('data', onData)
    req.on('end', () => resolve(data))
    req.on('error', reject)
  })
}

const LOGIN_PAGE = `<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Manager</title><style>
body{background:#0f1115;color:#e6e9ef;font:14px/1.5 'Segoe UI',system-ui,sans-serif;display:flex;
align-items:center;justify-content:center;height:100vh;margin:0}
form{background:#161920;border:1px solid #262b36;border-radius:8px;padding:28px;width:320px}
h1{font-size:16px;margin:0 0 4px}p{color:#98a1b3;font-size:12px;margin:0 0 18px}
input{width:100%;background:#0b0d11;border:1px solid #262b36;border-radius:5px;color:#e6e9ef;
padding:9px 10px;font:inherit;margin-bottom:12px}
button{width:100%;background:#2f4d80;border:1px solid #5b9dff;border-radius:5px;color:#e6e9ef;
padding:9px;font:inherit;cursor:pointer}
.err{color:#f87171;font-size:12px;margin-bottom:10px}
</style></head><body>
<form method="POST" action="/login">
<h1>LLM Manager</h1><p>__SUBTITLE__</p>
__ERROR__
<input type="password" name="password" placeholder="Password" autofocus autocomplete="current-password">
<button type="submit">Sign in</button>
</form></body></html>`

export class RemoteWebServer {
  private server: http.Server | https.Server | null = null
  private opts: WebServerOptions | null = null
  /** Connected SSE clients, mirroring the desktop's IPC event channel. */
  private clients = new Set<http.ServerResponse>()

  get running(): boolean {
    return !!this.server?.listening
  }

  /** Fan an event out to every connected remote browser. */
  broadcast(channel: string, payload: unknown): void {
    if (!this.clients.size) return
    const frame = `data: ${JSON.stringify({ channel, payload })}\n\n`
    for (const client of this.clients) {
      try {
        client.write(frame)
      } catch {
        this.clients.delete(client)
      }
    }
  }

  async start(opts: WebServerOptions): Promise<number> {
    if (!hasPassword()) {
      throw new Error('Set a password before enabling remote access.')
    }
    await this.stop()
    this.opts = opts

    const handler = (req: http.IncomingMessage, res: http.ServerResponse): void => {
      void this.handle(req, res).catch((err) => {
        res.writeHead(500, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }))
      })
    }

    const server = opts.tls
      ? https.createServer(
          { cert: fs.readFileSync(opts.tls.certPath), key: fs.readFileSync(opts.tls.keyPath) },
          handler
        )
      : http.createServer(handler)

    /*
     * Listen only as widely as the deployment needs.
     *
     * A tunnel is reached through cloudflared running on this machine, so loopback is enough —
     * binding every interface additionally published the UI to the local network, where the
     * spoofable-header problem above is reachable and where nobody asked for it to be. Serving a
     * domain directly does need every interface, because the client really is out there.
     */
    const host = opts.mode === 'own-domain' ? '0.0.0.0' : '127.0.0.1'

    await new Promise<void>((resolve, reject) => {
      server.once('error', reject)
      server.listen(opts.port, host, () => resolve())
    })
    this.server = server
    return opts.port
  }

  async stop(): Promise<void> {
    const server = this.server
    this.server = null
    if (!server) return
    await new Promise<void>((resolve) => server.close(() => resolve()))
  }

  private async handle(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`)
    // See WebServerOptions.mode: the header is only meaningful when Cloudflare set it.
    const forwarded = this.opts?.mode === 'tunnel' ? (req.headers['cf-connecting-ip'] as string | undefined) : undefined
    const ip = forwarded ?? req.socket.remoteAddress ?? 'unknown'
    const cookies = parseCookies(req.headers.cookie)
    const authed = verifySession(cookies.llmm_session)

    // Security headers on everything we serve.
    const baseHeaders = {
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY',
      'Referrer-Policy': 'no-referrer'
    }

    if (url.pathname === '/login' && req.method === 'POST') {
      const locked = isLockedOut(ip)
      if (locked > 0) {
        res.writeHead(429, { 'Content-Type': 'text/html', ...baseHeaders })
        res.end(loginPage(`Too many attempts. Try again in ${locked}s.`))
        return
      }
      const body = await readBody(req, 4096)
      const password = new URLSearchParams(body).get('password') ?? ''
      if (verifyPassword(password)) {
        recordSuccess(ip)
        const token = issueSession()
        res.writeHead(302, {
          Location: '/',
          'Set-Cookie': `llmm_session=${encodeURIComponent(token)}; HttpOnly; SameSite=Strict; Path=/; Max-Age=${7 * 24 * 3600}${this.opts?.tls ? '; Secure' : ''}`,
          ...baseHeaders
        })
        res.end()
        return
      }
      recordFailure(ip)
      res.writeHead(401, { 'Content-Type': 'text/html', ...baseHeaders })
      res.end(loginPage('Incorrect password.'))
      return
    }

    if (url.pathname === '/logout') {
      res.writeHead(302, { Location: '/', 'Set-Cookie': 'llmm_session=; Max-Age=0; Path=/', ...baseHeaders })
      res.end()
      return
    }

    if (!authed) {
      res.writeHead(url.pathname.startsWith('/bridge') ? 401 : 200, {
        'Content-Type': url.pathname.startsWith('/bridge') ? 'application/json' : 'text/html',
        ...baseHeaders
      })
      res.end(url.pathname.startsWith('/bridge') ? JSON.stringify({ error: 'not authenticated' }) : loginPage(null))
      return
    }

    // ---- authenticated from here on ----

    if (url.pathname === '/bridge' && req.method === 'POST') {
      const { channel, args } = JSON.parse(await readBody(req)) as { channel: string; args: unknown[] }

      const privileged = channel === 'settings:patch' ? privilegedSettingsIn((args ?? [])[0]) : null

      if (DESKTOP_ONLY.has(channel) || privileged) {
        res.writeHead(403, { 'Content-Type': 'application/json', ...baseHeaders })
        res.end(
          JSON.stringify({
            error: privileged
              ? `"${privileged}" must be changed on the desktop app. Security-related settings are not available remotely.`
              : `"${channel}" must be confirmed on the desktop app. Destructive and security-related actions are not available remotely.`
          })
        )
        return
      }

      try {
        const result = await this.opts!.invoke(channel, args ?? [])
        res.writeHead(200, { 'Content-Type': 'application/json', ...baseHeaders })
        res.end(JSON.stringify({ result }))
      } catch (err) {
        res.writeHead(200, { 'Content-Type': 'application/json', ...baseHeaders })
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }))
      }
      return
    }

    // Event stream: the remote equivalent of the desktop's IPC events.
    if (url.pathname === '/events') {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
        ...baseHeaders
      })
      res.write(': connected\n\n')
      this.clients.add(res)
      // A periodic comment keeps intermediaries from closing an idle stream.
      const keepAlive = setInterval(() => res.write(': ping\n\n'), 25000)
      req.on('close', () => {
        clearInterval(keepAlive)
        this.clients.delete(res)
      })
      return
    }

    /*
     * Attached files, for a transcript being read from a browser.
     *
     * The desktop shell reaches these over a registered scheme; this is the same resolver behind
     * a route, so the two shells show the same thing. Authentication has already been enforced
     * above — this sits below the gate deliberately, because these are the user's own files.
     *
     * The id is the only input, and it is looked up rather than joined onto anything, so unlike
     * the static branch below there is no containment test to get wrong.
     */
    if (url.pathname === '/media') {
      const id = url.searchParams.get('id') ?? ''
      const variant = url.searchParams.get('v') === 'optimised' ? 'optimised' : 'source'
      const hit = id ? resolveMedia(id, variant, req.headers.range ?? null) : null
      if (!hit) {
        res.writeHead(404, baseHeaders)
        res.end()
        return
      }
      res.writeHead(hit.range ? 206 : 200, { ...mediaHeaders(hit), ...baseHeaders })
      if (req.method === 'HEAD') {
        res.end()
        return
      }
      const stream = mediaStream(hit)
      // A player that seeks away mid-download aborts the response; without this the read stream
      // is left open holding a file handle for every scrub.
      res.on('close', () => stream.destroy())
      stream.on('error', () => res.destroy())
      stream.pipe(res)
      return
    }

    // Static renderer assets.
    // Resolved here so the containment test below compares two paths in the same normal form.
    const root = path.resolve(rendererRoot())
    const requested = url.pathname === '/' ? '/index.html' : url.pathname
    const filePath = path.resolve(root, `.${path.posix.normalize(requested)}`)

    /*
     * Containment is checked against a resolved path, with the separator included.
     *
     * A bare `startsWith(root)` is not a containment test: it also accepts a sibling whose name
     * merely begins with the root's, so a `dist-private` next to `dist` would serve. Nothing
     * reaches that today — pathname arrives percent-encoded, so `%2e%2e` never becomes `..` —
     * but this is the last check standing between the internet and the filesystem, and it
     * should not depend on that staying true.
     */
    if (filePath !== root && !filePath.startsWith(root + path.sep)) {
      res.writeHead(403, baseHeaders)
      res.end('Forbidden')
      return
    }

    if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
      const ext = path.extname(filePath)
      res.writeHead(200, { 'Content-Type': MIME[ext] ?? 'application/octet-stream', ...baseHeaders })
      fs.createReadStream(filePath).pipe(res)
      return
    }

    // SPA fallback.
    const index = path.join(root, 'index.html')
    if (fs.existsSync(index)) {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', ...baseHeaders })
      fs.createReadStream(index).pipe(res)
      return
    }

    res.writeHead(404, baseHeaders)
    res.end('Not found')
  }
}

function loginPage(error: string | null): string {
  return LOGIN_PAGE.replace('__SUBTITLE__', 'Sign in to reach this machine').replace(
    '__ERROR__',
    error ? `<div class="err">${error}</div>` : ''
  )
}

export const remoteWeb = new RemoteWebServer()
