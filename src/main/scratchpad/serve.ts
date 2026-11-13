/**
 * Loopback HTTP for what the scratchpad holds in memory, so ffmpeg can read it as if it were a file.
 *
 * ffmpeg and ffprobe take a URL wherever they take a path, and seek with range requests, which is
 * all a video needs: a frame near the end of an hour-long file is one ranged read, not an hour of
 * bytes, and an MP4 whose index sits at the end of the file opens like any other.
 *
 * Bound to 127.0.0.1 on a port the system picks, and every URL carries a random token, so nothing
 * else on the machine can guess its way in, and nothing off it can reach it at all. Started the
 * first time it is needed and stopped once nothing in memory is left to serve.
 */

import http from 'node:http'
import crypto from 'node:crypto'
import type { AddressInfo } from 'node:net'
import type { Readable } from 'node:stream'

export interface Served {
  sizeOf(id: string): number | null
  readRange(id: string, start: number, end: number): Readable | null
}

/** Parse one `Range: bytes=…` header against a file of `size` bytes; null when it is not satisfiable. */
export function parseRange(header: string | undefined, size: number): { start: number; end: number } | 'all' | null {
  if (!header) return 'all'
  const m = /^bytes=(\d*)-(\d*)$/.exec(header.trim())
  if (!m || (m[1] === '' && m[2] === '')) return null
  if (m[1] === '') {
    // The last N bytes.
    const n = Number(m[2])
    if (!n) return null
    return { start: Math.max(0, size - n), end: size - 1 }
  }
  const start = Number(m[1])
  const end = m[2] === '' ? size - 1 : Math.min(Number(m[2]), size - 1)
  if (start >= size || end < start) return null
  return { start, end }
}

export class ScratchServer {
  private server: http.Server | null = null
  private starting: Promise<void> | null = null
  private port = 0
  private readonly token = crypto.randomBytes(24).toString('hex')

  constructor(private readonly store: Served) {}

  /** A URL ffmpeg can read the file with, starting the server if it is not running. */
  async urlFor(id: string): Promise<string> {
    await this.start()
    return `http://127.0.0.1:${this.port}/${this.token}/${id}`
  }

  private start(): Promise<void> {
    if (this.server) return Promise.resolve()
    if (this.starting) return this.starting
    this.starting = new Promise<void>((resolve, reject) => {
      const server = http.createServer((req, res) => this.handle(req, res))
      server.on('error', reject)
      server.listen(0, '127.0.0.1', () => {
        this.port = (server.address() as AddressInfo).port
        this.server = server
        resolve()
      })
    }).finally(() => {
      this.starting = null
    })
    return this.starting
  }

  stop(): void {
    this.server?.close()
    this.server = null
  }

  get running(): boolean {
    return !!this.server
  }

  private handle(req: http.IncomingMessage, res: http.ServerResponse): void {
    const parts = (req.url ?? '').split('?')[0].split('/')
    // "/<token>/<id>", nothing more and nothing less.
    if (parts.length !== 3 || parts[1] !== this.token || !/^[0-9a-f]{32}$/.test(parts[2] ?? '')) {
      res.writeHead(404).end()
      return
    }
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.writeHead(405, { Allow: 'GET, HEAD' }).end()
      return
    }
    const id = parts[2]
    const size = this.store.sizeOf(id)
    if (size === null) {
      res.writeHead(404).end()
      return
    }
    const range = parseRange(req.headers.range, size)
    if (range === null) {
      res.writeHead(416, { 'Content-Range': `bytes */${size}` }).end()
      return
    }
    const { start, end } = range === 'all' ? { start: 0, end: size - 1 } : range
    const headers: http.OutgoingHttpHeaders = {
      'Accept-Ranges': 'bytes',
      'Content-Type': 'application/octet-stream',
      'Content-Length': size === 0 ? 0 : end - start + 1,
      'Cache-Control': 'no-store'
    }
    if (range !== 'all') headers['Content-Range'] = `bytes ${start}-${end}/${size}`
    res.writeHead(range === 'all' ? 200 : 206, headers)
    if (req.method === 'HEAD' || size === 0) {
      res.end()
      return
    }
    const body = this.store.readRange(id, start, end)
    if (!body) {
      res.destroy()
      return
    }
    // ffmpeg closes a connection the moment it has the bytes it wanted; that is not an error.
    body.on('error', () => res.destroy())
    res.on('close', () => body.destroy())
    body.pipe(res)
  }
}
