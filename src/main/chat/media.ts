/**
 * Serving attached files back to the transcript.
 *
 * A message used to say `[Attached: clip.mp4]` and nothing more, which is a strange thing for an
 * app that just spent real effort looking at the file. The bytes are still on disk, so the only
 * thing missing was a way for the renderer to ask for them.
 *
 * Two shells, one resolver. The desktop renderer is loaded from `file://` and reaches this over a
 * registered `llmm-media` scheme; a remote browser reaches the same function through a route on
 * the web server. Both arrive here, and neither of them can name a file.
 *
 * That last point is the whole security model. The only input is an attachment id, which is
 * looked up in the database; a client that invents an id gets nothing, and a client that asks for
 * `../../` gets nothing, because no part of the request is ever treated as a path. It matters
 * because the remote server is reachable from outside the machine.
 */

import fs from 'node:fs'
import path from 'node:path'
import { Readable } from 'node:stream'
import { attachmentFile } from './repo'

/** Which of the two files an attachment can have. */
export type MediaVariant = 'source' | 'optimised'

const MIME: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.bmp': 'image/bmp',
  '.svg': 'image/svg+xml',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.mkv': 'video/x-matroska',
  '.mov': 'video/quicktime',
  '.avi': 'video/x-msvideo',
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.flac': 'audio/flac',
  '.ogg': 'audio/ogg',
  '.m4a': 'audio/mp4'
}

export function mimeFor(file: string): string {
  return MIME[path.extname(file).toLowerCase()] ?? 'application/octet-stream'
}

export interface MediaHit {
  file: string
  size: number
  mime: string
  /** Present when the client asked for a byte range and it was satisfiable. */
  range?: { start: number; end: number }
}

/**
 * Turn an attachment id and a Range header into a file and a byte window.
 *
 * Range matters more than it looks. Without it a `<video>` element can still play, but seeking
 * means fetching the whole file again from the start — on a phone video that is hundreds of
 * megabytes through an IPC boundary every time the user drags the scrubber.
 *
 * Returns null for anything that cannot be served, deliberately without distinguishing "no such
 * attachment" from "the file moved": a caller that can tell those apart can probe for which ids
 * exist.
 */
export function resolveMedia(id: string, variant: MediaVariant, rangeHeader?: string | null): MediaHit | null {
  const file = attachmentFile(id, variant)
  if (!file) return null

  let size: number
  try {
    const stat = fs.statSync(file)
    if (!stat.isFile()) return null
    size = stat.size
  } catch {
    // Attached from a USB stick that has since been unplugged, or a temp file that was cleaned up.
    return null
  }

  const mime = mimeFor(file)
  const range = parseRange(rangeHeader ?? null, size)
  // An unsatisfiable range is not the same as no range: answering it with the whole file would
  // leave the player waiting for bytes at an offset it never receives.
  if (range === 'invalid') return null
  return range ? { file, size, mime, range } : { file, size, mime }
}

/**
 * The single open-ended byte range players actually send.
 *
 * Multi-range requests are legal and no media element issues them, so they are declined rather
 * than half-implemented.
 */
function parseRange(header: string | null, size: number): { start: number; end: number } | 'invalid' | null {
  if (!header) return null
  const m = /^bytes=(\d*)-(\d*)$/.exec(header.trim())
  if (!m) return null
  if (size === 0) return 'invalid'

  const [, rawStart, rawEnd] = m
  let start: number
  let end: number
  if (rawStart === '') {
    // A suffix range: the last N bytes.
    const len = Number(rawEnd)
    if (!Number.isFinite(len) || len <= 0) return 'invalid'
    start = Math.max(0, size - len)
    end = size - 1
  } else {
    start = Number(rawStart)
    end = rawEnd === '' ? size - 1 : Number(rawEnd)
  }

  if (!Number.isFinite(start) || !Number.isFinite(end) || start > end || start >= size) return 'invalid'
  return { start, end: Math.min(end, size - 1) }
}

/** Headers describing a hit, including the ones that let a player seek. */
export function mediaHeaders(hit: MediaHit): Record<string, string> {
  const length = hit.range ? hit.range.end - hit.range.start + 1 : hit.size
  const headers: Record<string, string> = {
    'Content-Type': hit.mime,
    'Content-Length': String(length),
    'Accept-Ranges': 'bytes',
    // Local files that never change identity, addressed by an id that is minted per attachment.
    'Cache-Control': 'private, max-age=3600'
  }
  if (hit.range) headers['Content-Range'] = `bytes ${hit.range.start}-${hit.range.end}/${hit.size}`
  return headers
}

export function mediaStream(hit: MediaHit): fs.ReadStream {
  return hit.range
    ? fs.createReadStream(hit.file, { start: hit.range.start, end: hit.range.end })
    : fs.createReadStream(hit.file)
}

/**
 * Answer a `llmm-media://` request from the desktop renderer.
 *
 * The URL carries the id in its path and the variant in its query, so nothing about it is
 * position-dependent or guessable into a path.
 */
export function handleMediaProtocol(request: Request): Response {
  let url: URL
  try {
    url = new URL(request.url)
  } catch {
    return new Response(null, { status: 400 })
  }

  const id = url.pathname.replace(/^\/+/, '')
  const variant: MediaVariant = url.searchParams.get('v') === 'optimised' ? 'optimised' : 'source'
  const hit = id ? resolveMedia(id, variant, request.headers.get('range')) : null
  if (!hit) return new Response(null, { status: 404 })

  // Readable.toWeb hands Chromium a real stream, so a large file is never held in memory whole.
  const body = Readable.toWeb(mediaStream(hit)) as ReadableStream
  return new Response(body, { status: hit.range ? 206 : 200, headers: mediaHeaders(hit) })
}
