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

class DownloadQueue extends EventEmitter {
  private active = new Map<string, AbortController>()
  private speeds = new Map<string, number>()
  private token: string | null = null

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

  private async download(item: DownloadItem): Promise<void> {
    const controller = new AbortController()
    this.active.set(item.id, controller)
    this.setStatus(item.id, 'downloading')

    const partial = this.partialPath(item.dest)
    await fsp.mkdir(path.dirname(partial), { recursive: true })
    await fsp.mkdir(path.dirname(item.dest), { recursive: true })

    let startAt = 0
    try {
      startAt = (await fsp.stat(partial)).size
    } catch {
      startAt = 0
    }

    try {
      const headers: Record<string, string> = {}
      if (this.token) headers.Authorization = `Bearer ${this.token}`
      // The resume path: ask only for what we do not already have.
      if (startAt > 0) headers.Range = `bytes=${startAt}-`

      const res = await fetch(item.url, { headers, signal: controller.signal, redirect: 'follow' })

      if (res.status === 416) {
        // Range not satisfiable: the partial is already the whole file.
        await this.finalise(item, partial)
        return
      }
      if (!res.ok && res.status !== 206) {
        throw new Error(`HTTP ${res.status} ${res.statusText}`)
      }
      if (startAt > 0 && res.status !== 206) {
        // Server ignored the range; start over rather than corrupting the file.
        startAt = 0
        await fsp.rm(partial, { force: true })
      }
      if (!res.body) throw new Error('No response body')

      const contentLength = Number(res.headers.get('content-length') ?? 0)
      const total = startAt + contentLength
      if (total > 0 && total !== item.bytesTotal) {
        run('UPDATE downloads SET bytes_total = ? WHERE id = ?', total, item.id)
      }

      let done = startAt
      let lastTick = Date.now()
      let lastBytes = done

      const out = fs.createWriteStream(partial, { flags: startAt > 0 ? 'a' : 'w' })
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

      /*
       * A short file is a failure, not a finished download.
       *
       * Nothing checked the length before renaming, so a connection dropped near the end — or a
       * proxy that closed the body early — produced a truncated .gguf sitting in the models
       * folder under its real name, indistinguishable from a good one until something tried to
       * load it. Left as a .partial, it resumes from where it stopped instead.
       */
      // Only when the server said how long the body would be. Without Content-Length, `total`
      // collapses to whatever the resume started at, and every good chunked resume looks short.
      if (contentLength > 0 && done !== total) {
        throw new Error(
          `Download stopped at ${done} of ${total} bytes. The partial file was kept, so resuming will continue from there.`
        )
      }

      await this.finalise(item, partial)
    } catch (err) {
      this.active.delete(item.id)
      this.speeds.delete(item.id)
      if (controller.signal.aborted) {
        // Paused or cancelled: the partial file stays for a later resume.
        void this.pump()
        return
      }
      this.setStatus(item.id, 'failed', err instanceof Error ? err.message : String(err))
      void this.pump()
    }
  }

  private async finalise(item: DownloadItem, partial: string): Promise<void> {
    await fsp.rename(partial, item.dest)
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

    const known = new Set(
      all<Row>("SELECT * FROM downloads WHERE status IN ('queued','paused','downloading')").map((r) =>
        path.basename(this.partialPath(r.dest))
      )
    )

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
