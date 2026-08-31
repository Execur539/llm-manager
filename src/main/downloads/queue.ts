/**
 * Download queue.
 *
 * Resumable by necessity: these files are routinely 20 GB, and a download that has to restart
 * from zero because the app was closed is not acceptable. Partial files live in
 * `<models>/.partial` and are completed with HTTP range requests, so a restart picks up where
 * it stopped rather than beginning again.
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import crypto from 'node:crypto'
import { EventEmitter } from 'node:events'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { all, get, run } from '../storage/db'

export type DownloadStatus = 'queued' | 'downloading' | 'paused' | 'done' | 'failed' | 'cancelled'

export interface DownloadItem {
  id: string
  repo: string | null
  filename: string
  url: string
  dest: string
  bytesTotal: number
  bytesDone: number
  status: DownloadStatus
  error: string | null
  /** bytes per second over the last sample window */
  speed: number
}

interface Row {
  id: string
  repo: string | null
  filename: string
  url: string
  dest: string
  bytes_total: number
  bytes_done: number
  status: string
  error: string | null
}

const toItem = (r: Row): DownloadItem => ({
  id: r.id,
  repo: r.repo,
  filename: r.filename,
  url: r.url,
  dest: r.dest,
  bytesTotal: r.bytes_total,
  bytesDone: r.bytes_done,
  status: r.status as DownloadStatus,
  error: r.error,
  speed: 0
})

/** How many files transfer at once. More than a couple just splits the same bandwidth. */
const MAX_CONCURRENT = 2

/**
 * Connections used for one file.
 *
 * Whether this helps depends entirely on where the bottleneck is. Measured against HuggingFace's
 * CDN on a ~90 Mbit line, one connection and eight were within 4% of each other — the line was
 * saturated either way, and splitting it changed nothing. On a faster link, where a single
 * connection is limited by the server's per-stream shaping or by the bandwidth-delay product
 * rather than by the pipe, several ranges in flight is the difference between a fraction of the
 * line and all of it. HuggingFace ship `hf_transfer` for exactly this reason.
 *
 * So it is on, at a modest number: it costs nothing measurable when the line is the limit, and
 * it is the whole win when it is not.
 */
const DEFAULT_CONNECTIONS = 4

/**
 * Smallest slice worth giving its own connection.
 *
 * Splitting a 20 MB file eight ways means eight requests, eight TLS handshakes and eight sets of
 * slow-start, to move two and a half megabytes each. The part count is reduced until each part
 * is at least this large, so small files quietly use one connection.
 */
const MIN_PART_BYTES = 8 * 1024 * 1024

/** Write buffer per part. The default 64 KB means far more syscalls than a bulk transfer needs. */
const WRITE_CHUNK = 4 * 1024 * 1024

/** Transient failures worth retrying before giving up and waiting for the user. */
const MAX_ATTEMPTS = 5

/** One contiguous slice of the file, and how much of it has landed. */
interface PartState {
  start: number
  end: number
  done: number
}

/**
 * Resume state for a multi-part download, kept beside the .partial file.
 *
 * A single-stream download can resume from the size of what it has, because the bytes are
 * contiguous. Parts are not: the file has holes while it is in flight, so its size says nothing
 * about which ranges are present. This records that, and is rewritten as the parts advance.
 */
interface PartsFile {
  url: string
  total: number
  parts: PartState[]
}

class DownloadQueue extends EventEmitter {
  private active = new Map<string, AbortController>()
  private speeds = new Map<string, number>()
  private token: string | null = null
  /** Connections per file; see DEFAULT_CONNECTIONS for why this is worth tuning. */
  private connections = DEFAULT_CONNECTIONS

  setConnections(n: number): void {
    this.connections = Math.max(1, Math.min(16, Math.round(n) || DEFAULT_CONNECTIONS))
  }

  setToken(token: string | null): void {
    this.token = token
  }

  list(): DownloadItem[] {
    return all<Row>('SELECT * FROM downloads ORDER BY created_at DESC').map((r) => ({
      ...toItem(r),
      speed: this.speeds.get(r.id) ?? 0
    }))
  }

  enqueue(opts: { repo: string | null; filename: string; url: string; dest: string; bytesTotal: number }): DownloadItem {
    const id = crypto.randomBytes(6).toString('hex')
    const now = Date.now()
    run(
      'INSERT INTO downloads (id, repo, filename, url, dest, bytes_total, bytes_done, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)',
      id,
      opts.repo,
      opts.filename,
      opts.url,
      opts.dest,
      opts.bytesTotal,
      'queued',
      now,
      now
    )
    this.emit('update', this.list())
    void this.pump()
    return {
      id,
      repo: opts.repo,
      filename: opts.filename,
      url: opts.url,
      dest: opts.dest,
      bytesTotal: opts.bytesTotal,
      bytesDone: 0,
      status: 'queued',
      error: null,
      speed: 0
    }
  }

  private setStatus(id: string, status: DownloadStatus, error?: string): void {
    run('UPDATE downloads SET status = ?, error = ?, updated_at = ? WHERE id = ?', status, error ?? null, Date.now(), id)
    this.emit('update', this.list())
  }

  private setProgress(id: string, bytesDone: number): void {
    run('UPDATE downloads SET bytes_done = ?, updated_at = ? WHERE id = ?', bytesDone, Date.now(), id)
  }

  pause(id: string): void {
    this.active.get(id)?.abort()
    this.active.delete(id)
    this.setStatus(id, 'paused')
  }

  resume(id: string): void {
    this.setStatus(id, 'queued')
    void this.pump()
  }

  async cancel(id: string): Promise<void> {
    this.active.get(id)?.abort()
    this.active.delete(id)
    const row = get<Row>('SELECT * FROM downloads WHERE id = ?', id)
    if (row) {
      const partial = this.partialPath(row.dest)
      await fsp.rm(partial, { force: true }).catch(() => undefined)
      await fsp.rm(`${partial}.parts`, { force: true }).catch(() => undefined)
    }
    this.setStatus(id, 'cancelled')
  }

  remove(id: string): void {
    this.active.get(id)?.abort()
    this.active.delete(id)
    run('DELETE FROM downloads WHERE id = ?', id)
    this.emit('update', this.list())
  }

  private partialPath(dest: string): string {
    const dir = path.join(path.dirname(dest), '.partial')
    return path.join(dir, `${path.basename(dest)}.part`)
  }

  /** Start queued items up to the concurrency limit. */
  private async pump(): Promise<void> {
    if (this.active.size >= MAX_CONCURRENT) return
    const queued = all<Row>("SELECT * FROM downloads WHERE status = 'queued' ORDER BY created_at").slice(
      0,
      MAX_CONCURRENT - this.active.size
    )
    for (const row of queued) {
      void this.download(toItem(row))
    }
  }

  /**
   * Run one download, retrying transient failures rather than parking it.
   *
   * A twenty-gigabyte transfer over half an hour will meet a dropped connection, a DNS blip or a
   * CDN hiccup sooner or later. The partial was always kept, so resuming worked — but only if
   * somebody noticed the row had gone red and clicked Resume, which on a download left running
   * overnight means the morning is spent discovering it stopped at 90% and doing it again.
   *
   * A pause or a cancel is not a failure and never retries; the abort signal distinguishes them.
   */
  private async download(item: DownloadItem): Promise<void> {
    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      const outcome = await this.attempt(item, attempt)
      if (outcome.settled) return

      // Back off, but stay responsive to a cancel arriving mid-wait.
      const delay = Math.min(30_000, 2 ** (attempt - 1) * 1000)
      this.setStatus(item.id, 'downloading', `${outcome.error} — retrying in ${Math.round(delay / 1000)}s (${attempt}/${MAX_ATTEMPTS})`)
      const cancelled = await this.sleep(delay, item.id)
      if (cancelled) return
    }

    this.setStatus(item.id, 'failed', `Gave up after ${MAX_ATTEMPTS} attempts. The partial file was kept, so Resume continues from where it stopped.`)
    void this.pump()
  }

  /** Wait, unless the download is cancelled first. Resolves true when it was. */
  private sleep(ms: number, id: string): Promise<boolean> {
    return new Promise((resolve) => {
      const timer = setTimeout(() => resolve(false), ms)
      const controller = this.active.get(id)
      const onAbort = (): void => {
        clearTimeout(timer)
        resolve(true)
      }
      if (controller?.signal.aborted) onAbort()
      else controller?.signal.addEventListener('abort', onAbort, { once: true })
    })
  }

  private async attempt(item: DownloadItem, attemptNumber: number): Promise<{ settled: boolean; error: string }> {
    const controller = new AbortController()
    this.active.set(item.id, controller)
    if (attemptNumber === 1) this.setStatus(item.id, 'downloading')

    const partial = this.partialPath(item.dest)
    const partsFile = `${partial}.parts`
    await fsp.mkdir(path.dirname(partial), { recursive: true })
    await fsp.mkdir(path.dirname(item.dest), { recursive: true })

    const headers: Record<string, string> = {}
    if (this.token) headers.Authorization = `Bearer ${this.token}`

    try {
      const plan = await this.planTransfer(item, partial, partsFile, headers, controller.signal)

      if (plan.alreadyComplete) {
        await this.finalise(item, partial, partsFile)
        return { settled: true, error: '' }
      }

      if (plan.total > 0 && plan.total !== item.bytesTotal) {
        run('UPDATE downloads SET bytes_total = ? WHERE id = ?', plan.total, item.id)
      }

      const done = plan.parts
        ? await this.transferParts(item, partial, partsFile, plan, headers, controller.signal)
        : await this.transferWhole(item, partial, plan, headers, controller.signal)

      /*
       * A short file is a failure, not a finished download.
       *
       * Nothing checked the length before renaming, so a connection dropped near the end — or a
       * proxy that closed the body early — produced a truncated .gguf sitting in the models
       * folder under its real name, indistinguishable from a good one until something tried to
       * load it. Left as a .partial, it resumes from where it stopped instead.
       */
      if (plan.total > 0 && done !== plan.total) {
        throw new Error(`stopped at ${done} of ${plan.total} bytes`)
      }

      await this.finalise(item, partial, partsFile)
      return { settled: true, error: '' }
    } catch (err) {
      this.active.delete(item.id)
      this.speeds.delete(item.id)

      if (controller.signal.aborted) {
        // Paused or cancelled: the partial stays for a later resume, and this is not a failure.
        void this.pump()
        return { settled: true, error: '' }
      }

      const message = err instanceof Error ? err.message : String(err)

      // A refusal will be refused again. Only conditions that might pass on a second look are
      // worth retrying; anything else parks immediately so the user sees the real reason.
      if (/HTTP (4[0-9]{2})/.test(message) && !/HTTP (408|425|429)/.test(message)) {
        this.setStatus(item.id, 'failed', message)
        void this.pump()
        return { settled: true, error: message }
      }
      return { settled: false, error: message }
    }
  }

  /**
   * Decide how to fetch this file, and pick up whatever a previous attempt left behind.
   *
   * Parts are only used when the server offers ranges and the file is big enough to be worth
   * splitting. Everything else falls back to the single stream, which is also the path for a
   * server that answers a range request with the whole file.
   */
  private async planTransfer(
    item: DownloadItem,
    partial: string,
    partsFile: string,
    headers: Record<string, string>,
    signal: AbortSignal
  ): Promise<{ total: number; parts: PartState[] | null; resumeAt: number; alreadyComplete: boolean }> {
    // A parts file from an earlier attempt is authoritative about what is already on disk.
    const saved = await fsp
      .readFile(partsFile, 'utf8')
      .then((raw) => JSON.parse(raw) as PartsFile)
      .catch(() => null)

    if (saved && saved.url === item.url && saved.parts.length) {
      const have = saved.parts.reduce((a, p) => a + p.done, 0)
      if (have >= saved.total) return { total: saved.total, parts: saved.parts, resumeAt: have, alreadyComplete: true }
      return { total: saved.total, parts: saved.parts, resumeAt: have, alreadyComplete: false }
    }

    // No parts file: either a fresh download or a single-stream one to continue.
    const contiguous = await fsp
      .stat(partial)
      .then((s) => s.size)
      .catch(() => 0)

    const probe = await fetch(item.url, {
      method: 'HEAD',
      headers,
      signal,
      redirect: 'follow'
    }).catch(() => null)

    const total = Number(probe?.headers.get('content-length') ?? 0) || item.bytesTotal || 0
    const acceptsRanges = probe?.headers.get('accept-ranges')?.toLowerCase().includes('bytes') ?? false

    if (total > 0 && contiguous >= total) {
      return { total, parts: null, resumeAt: contiguous, alreadyComplete: true }
    }

    /*
     * Splitting is only worth it on a file large enough that the extra handshakes disappear
     * into the transfer, and only when there is nothing already downloaded contiguously — a
     * half-finished single-stream download is cheaper to continue than to restart in parts.
     */
    const connections = Math.max(1, Math.min(16, this.connections))
    const usable = total > 0 && acceptsRanges && contiguous === 0 && connections > 1
    const partCount = usable ? Math.max(1, Math.min(connections, Math.floor(total / MIN_PART_BYTES))) : 1

    if (partCount <= 1) return { total, parts: null, resumeAt: contiguous, alreadyComplete: false }

    const size = Math.ceil(total / partCount)
    const parts: PartState[] = Array.from({ length: partCount }, (_, i) => ({
      start: i * size,
      end: Math.min(total, (i + 1) * size) - 1,
      done: 0
    }))

    // The file is created at full length up front so each part can write straight to its offset.
    const handle = await fsp.open(partial, 'w')
    await handle.truncate(total).catch(() => undefined)
    await handle.close()
    await fsp.writeFile(partsFile, JSON.stringify({ url: item.url, total, parts } satisfies PartsFile))

    return { total, parts, resumeAt: 0, alreadyComplete: false }
  }

  /** Several ranges at once, each writing straight into its own region of the file. */
  private async transferParts(
    item: DownloadItem,
    partial: string,
    partsFile: string,
    plan: { total: number; parts: PartState[] | null },
    headers: Record<string, string>,
    signal: AbortSignal
  ): Promise<number> {
    const parts = plan.parts as PartState[]
    const totalDone = (): number => parts.reduce((a, p) => a + p.done, 0)

    let lastTick = Date.now()
    let lastBytes = totalDone()
    let lastPersist = Date.now()

    const report = async (): Promise<void> => {
      const now = Date.now()
      if (now - lastTick < 500) return
      const done = totalDone()
      this.speeds.set(item.id, ((done - lastBytes) / (now - lastTick)) * 1000)
      lastBytes = done
      lastTick = now
      this.setProgress(item.id, done)
      this.emit('update', this.list())

      // The resume record is written less often than progress; losing a second of it on a crash
      // costs a second of re-download, while writing it per chunk would cost real throughput.
      if (now - lastPersist >= 2000) {
        lastPersist = now
        await fsp
          .writeFile(partsFile, JSON.stringify({ url: item.url, total: plan.total, parts } satisfies PartsFile))
          .catch(() => undefined)
      }
    }

    await Promise.all(
      parts.map(async (part) => {
        if (part.done > part.end - part.start) return

        const from = part.start + part.done
        if (from > part.end) return

        const res = await fetch(item.url, {
          headers: { ...headers, Range: `bytes=${from}-${part.end}` },
          signal,
          redirect: 'follow'
        })

        // A server that ignores the range would send the whole file down every part.
        if (res.status !== 206) throw new Error(`HTTP ${res.status} (expected a partial response)`)
        if (!res.body) throw new Error('No response body')

        const handle = await fsp.open(partial, 'r+')
        try {
          let position = from
          for await (const chunk of res.body as unknown as AsyncIterable<Uint8Array>) {
            await handle.write(chunk, 0, chunk.length, position)
            position += chunk.length
            part.done += chunk.length
            await report()
          }
        } finally {
          await handle.close()
        }
      })
    )

    const done = totalDone()
    this.setProgress(item.id, done)
    await fsp
      .writeFile(partsFile, JSON.stringify({ url: item.url, total: plan.total, parts } satisfies PartsFile))
      .catch(() => undefined)
    return done
  }

  /** One connection, bytes appended in order — for servers without ranges, and small files. */
  private async transferWhole(
    item: DownloadItem,
    partial: string,
    plan: { total: number; resumeAt: number },
    headers: Record<string, string>,
    signal: AbortSignal
  ): Promise<number> {
    let startAt = plan.resumeAt
    const requestHeaders = { ...headers }
    if (startAt > 0) requestHeaders.Range = `bytes=${startAt}-`

    const res = await fetch(item.url, { headers: requestHeaders, signal, redirect: 'follow' })

    if (res.status === 416) return startAt // the partial is already the whole file
    if (!res.ok && res.status !== 206) throw new Error(`HTTP ${res.status} ${res.statusText}`)
    if (startAt > 0 && res.status !== 206) {
      // Server ignored the range; start over rather than corrupting the file.
      startAt = 0
      await fsp.rm(partial, { force: true })
    }
    if (!res.body) throw new Error('No response body')

    let done = startAt
    let lastTick = Date.now()
    let lastBytes = done

    // A larger buffer than the 64 KB default: this is a bulk transfer, not a chat stream.
    const out = fs.createWriteStream(partial, { flags: startAt > 0 ? 'a' : 'w', highWaterMark: WRITE_CHUNK })
    const source = Readable.fromWeb(res.body as Parameters<typeof Readable.fromWeb>[0])

    source.on('data', (chunk: Buffer) => {
      done += chunk.length
      const now = Date.now()
      if (now - lastTick >= 500) {
        this.speeds.set(item.id, ((done - lastBytes) / (now - lastTick)) * 1000)
        lastBytes = done
        lastTick = now
        this.setProgress(item.id, done)
        this.emit('update', this.list())
      }
    })

    await pipeline(source, out)
    this.setProgress(item.id, done)
    return done
  }

  private async finalise(item: DownloadItem, partial: string, partsFile?: string): Promise<void> {
    await fsp.rename(partial, item.dest)
    // The resume record only describes a transfer in progress; a finished file must not leave
    // one behind for a later download of the same name to pick up.
    if (partsFile) await fsp.rm(partsFile, { force: true }).catch(() => undefined)
    this.active.delete(item.id)
    this.speeds.delete(item.id)
    this.setStatus(item.id, 'done')
    this.emit('completed', item)
    void this.pump()
  }

  /** On launch, mark anything that was mid-flight as paused so it can be resumed deliberately. */
  recoverOnStart(): void {
    run("UPDATE downloads SET status = 'paused' WHERE status = 'downloading'")
  }

  /** Delete orphaned .partial files with no matching queue entry. */
  async cleanPartials(modelsDir: string): Promise<{ removed: number; bytes: number }> {
    const partialDir = path.join(modelsDir, '.partial')
    if (!fs.existsSync(partialDir)) return { removed: 0, bytes: 0 }

    const live = all<Row>("SELECT * FROM downloads WHERE status IN ('queued','paused','downloading')").map((r) =>
      path.basename(this.partialPath(r.dest))
    )
    // A resumable download owns two files now. Sweeping the sidecar would not lose the bytes, but
    // it would lose the record of which ranges are present, forcing the whole file again.
    const known = new Set([...live, ...live.map((n) => `${n}.parts`)])

    let removed = 0
    let bytes = 0
    for (const name of await fsp.readdir(partialDir)) {
      if (known.has(name)) continue
      const full = path.join(partialDir, name)
      try {
        bytes += (await fsp.stat(full)).size
        await fsp.rm(full, { force: true })
        removed++
      } catch {
        /* best effort */
      }
    }
    return { removed, bytes }
  }
}

export const downloadQueue = new DownloadQueue()
