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
  'model:delete',
  'library:delete-model',
  'settings:set-password',
  'settings:disable-hard-blocks',
  'remote:disable',
  'remote:enable',
  'agent:set-hard-blocks',
  'mcp:add',
  'mcp:remove'
])

export interface WebServerOptions {
  port: number
  /** invoke a bridge handler by name; the same map the desktop IPC uses */
  invoke: (channel: string, args: unknown[]) => Promise<unknown>
  /** TLS material for the own-domain path; omitted means plain HTTP behind a tunnel */
  tls?: { certPath: string; keyPath: string }
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

function readBody(req: http.IncomingMessage, limit = 32 * 1024 * 1024): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = ''
    req.on('data', (c: Buffer) => {
      data += c
      if (data.length > limit) reject(new Error('body too large'))
    })
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

    await new Promise<void>((resolve, reject) => {
      server.once('error', reject)
      server.listen(opts.port, '0.0.0.0', () => resolve())
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
    const ip = (req.headers['cf-connecting-ip'] as string) ?? req.socket.remoteAddress ?? 'unknown'
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

      if (DESKTOP_ONLY.has(channel)) {
        res.writeHead(403, { 'Content-Type': 'application/json', ...baseHeaders })
        res.end(
          JSON.stringify({
            error: `"${channel}" must be confirmed on the desktop app. Destructive and security-related actions are not available remotely.`
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

    // Static renderer assets.
    const root = rendererRoot()
    const requested = url.pathname === '/' ? '/index.html' : url.pathname
    const filePath = path.join(root, path.normalize(requested).replace(/^(\.\.[/\\])+/, ''))

    if (!filePath.startsWith(root)) {
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
