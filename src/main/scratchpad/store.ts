/**
 * Chat's scratchpad: the only place a chat's model can keep anything.
 *
 * Chat has no access to the user's machine, unlike the agent. What it is given to work on, such as
 * an attached video, is put here under a plain name, and the tools it has (watch_video and the
 * scratchpad_* tools) can reach nothing else. Nothing here is meant to last:
 *
 * - By default it lives in memory, in blocks of a few megabytes, so it never touches a disk and
 *   is gone when the app closes, crash or not. Settings can point it at a folder instead.
 * - The model deletes a file when it is done with it.
 * - A file the model has not read or written for a while (Settings) is deleted anyway.
 *
 * Each conversation has its own. A file that is gone leaves a note saying why, so a model that
 * reaches for it later is told what happened rather than that it never existed.
 *
 * No Electron in here, so the store can be tested on its own; scratchpad/index.ts wires it to
 * settings, a sweep timer and the app's lifetime.
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import crypto from 'node:crypto'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'

export type ScratchKind = 'video' | 'text' | 'file'

export interface ScratchItem {
  /** Opaque, unguessable: used in loopback URLs. */
  id: string
  chatId: string
  /** What the model calls it; unique within its conversation, compared without case. */
  name: string
  kind: ScratchKind
  bytes: number
  createdAt: number
  /** The last time the model read or wrote it, or when it arrived. */
  usedAt: number
}

export interface ScratchSummary extends Omit<ScratchItem, 'id'> {
  /** Where it is held: 'memory', or the file's path in the chosen folder. */
  where: string
}

interface Entry extends ScratchItem {
  /** In memory: the bytes, in blocks. */
  blocks?: Buffer[]
  /** In a folder: the file. */
  file?: string
}

/** Large enough that a gigabyte is a few hundred blocks, small enough never to be a problem to allocate. */
const BLOCK = 8 * 1024 * 1024

/** The most text one scratchpad_write may hold. */
export const MAX_TEXT_BYTES = 1024 * 1024

export class ScratchpadError extends Error {}

/**
 * A name the model or the user gave, made safe to hold and to put on disk.
 *
 * No folders (a name is never a path), nothing Windows refuses in a file name, no leading dots or
 * trailing spaces, and not too long. What is left of an empty name becomes "file".
 */
export function safeName(raw: string): string {
  let name = path.basename(String(raw ?? '').replace(/\\/g, '/'))
  name = name.replace(/[<>:"/\\|?*\u0000-\u001f]/g, '_').replace(/^[.\s]+/, '').replace(/[.\s]+$/, '')
  if (/^(con|prn|aux|nul|com\d|lpt\d)(\..*)?$/i.test(name)) name = `_${name}`
  if (name.length > 120) {
    const ext = path.extname(name).slice(0, 16)
    name = name.slice(0, 120 - ext.length) + ext
  }
  return name || 'file'
}

/** A conversation id as a folder name: kept when it is plain, hashed when it is not. */
function folderFor(chatId: string): string {
  return /^[A-Za-z0-9_-]{1,64}$/.test(chatId) ? chatId : crypto.createHash('sha256').update(chatId).digest('hex').slice(0, 24)
}

export interface ScratchpadOptions {
  /** Where files go when they are not kept in memory; null keeps them in memory. Read on every add. */
  folder: (chatId?: string) => string | null
  /** The most a new file may take in memory, given what the system has free; past it, adding refuses. */
  memoryLimit: () => number
  now?: () => number
}

/** What the folder the app owns inside a chosen location is called. Only this is ever cleared. */
export const OWNED_FOLDER = 'LLM Manager scratchpad'

export class Scratchpad {
  private chats = new Map<string, Map<string, Entry>>()
  private byId = new Map<string, Entry>()
  private gone = new Map<string, Map<string, string>>()
  private readonly now: () => number

  constructor(private readonly opts: ScratchpadOptions) {
    this.now = opts.now ?? Date.now
  }

  private items(chatId: string): Map<string, Entry> {
    let m = this.chats.get(chatId)
    if (!m) {
      m = new Map()
      this.chats.set(chatId, m)
    }
    return m
  }

  /** A name not yet taken in this conversation: "clip.mp4", then "clip (2).mp4". */
  private freeName(chatId: string, wanted: string): string {
    const taken = this.items(chatId)
    const base = safeName(wanted)
    if (!taken.has(base.toLowerCase())) return base
    const ext = path.extname(base)
    const stem = base.slice(0, base.length - ext.length)
    for (let n = 2; ; n++) {
      const candidate = `${stem} (${n})${ext}`
      if (!taken.has(candidate.toLowerCase())) return candidate
    }
  }

  private ownedRoot(chatId?: string): string | null {
    const folder = this.opts.folder(chatId)
    return folder ? path.join(folder, OWNED_FOLDER) : null
  }

  /**
   * Put something in a conversation's scratchpad, from a stream, under a free name like `wanted`.
   *
   * `expectedBytes`, when known, is checked against what memory can spare before anything is
   * read, so a file too large to hold is refused with a reason instead of half-loaded.
   */
  async add(chatId: string, wanted: string, source: Readable, kind: ScratchKind, expectedBytes?: number): Promise<ScratchItem> {
    const name = this.freeName(chatId, wanted)
    const root = this.ownedRoot(chatId)
    const at = this.now()
    const entry: Entry = { id: crypto.randomBytes(16).toString('hex'), chatId, name, kind, bytes: 0, createdAt: at, usedAt: at }

    if (root) {
      const dir = path.join(root, folderFor(chatId))
      await fsp.mkdir(dir, { recursive: true })
      entry.file = path.join(dir, name)
      try {
        await pipeline(source, fs.createWriteStream(entry.file))
      } catch (err) {
        await fsp.rm(entry.file, { force: true }).catch(() => undefined)
        throw err
      }
      entry.bytes = (await fsp.stat(entry.file)).size
    } else {
      const limit = this.opts.memoryLimit()
      if (expectedBytes !== undefined && expectedBytes > limit) {
        source.destroy()
        throw new ScratchpadError(
          `${name} is ${gb(expectedBytes)}, more than the ${gb(limit)} of memory the scratchpad can use right now. Settings → Scratchpad can keep it in a folder instead.`
        )
      }
      const blocks: Buffer[] = []
      let current = Buffer.allocUnsafe(Math.min(BLOCK, Math.max(1, expectedBytes ?? BLOCK)))
      let fill = 0
      let total = 0
      for await (const chunk of source as AsyncIterable<Buffer | string>) {
        let buf = typeof chunk === 'string' ? Buffer.from(chunk) : chunk
        total += buf.length
        if (total > limit) {
          source.destroy()
          throw new ScratchpadError(
            `${name} is larger than the ${gb(limit)} of memory the scratchpad can use right now. Settings → Scratchpad can keep it in a folder instead.`
          )
        }
        while (buf.length) {
          if (fill === current.length) {
            blocks.push(current)
            current = Buffer.allocUnsafe(BLOCK)
            fill = 0
          }
          const n = Math.min(buf.length, current.length - fill)
          buf.copy(current, fill, 0, n)
          fill += n
          buf = buf.subarray(n)
        }
      }
      // The last block is copied to its size: a slice would keep the whole block it came from alive.
      if (fill) blocks.push(fill === current.length ? current : Buffer.from(current.subarray(0, fill)))
      entry.blocks = blocks
      entry.bytes = total
    }

    this.items(chatId).set(name.toLowerCase(), entry)
    this.byId.set(entry.id, entry)
    this.gone.get(chatId)?.delete(name.toLowerCase())
    return publicItem(entry)
  }

  /** Copy a file the user attached into the scratchpad. */
  async addFile(chatId: string, file: string, kind: ScratchKind): Promise<ScratchItem> {
    const { size } = await fsp.stat(file)
    return this.add(chatId, path.basename(file), fs.createReadStream(file), kind, size)
  }

  get(chatId: string, name: string): ScratchItem | undefined {
    const e = this.chats.get(chatId)?.get(safeName(name).toLowerCase()) ?? this.chats.get(chatId)?.get(String(name).toLowerCase())
    return e ? publicItem(e) : undefined
  }

  /** Mark a file as just read or written, which is what keeps it from being swept. */
  touch(item: ScratchItem): void {
    const e = this.byId.get(item.id)
    if (e) e.usedAt = this.now()
  }

  list(chatId: string): ScratchSummary[] {
    return [...(this.chats.get(chatId)?.values() ?? [])]
      .sort((a, b) => a.createdAt - b.createdAt)
      .map((e) => ({ ...publicItem(e), where: e.file ?? 'memory' }))
  }

  /** Every conversation with something in its scratchpad, and how much. */
  totals(): { files: number; bytes: number; inMemory: number } {
    let files = 0
    let bytes = 0
    let inMemory = 0
    for (const e of this.byId.values()) {
      files++
      bytes += e.bytes
      if (e.blocks) inMemory += e.bytes
    }
    return { files, bytes, inMemory }
  }

  /** Why a name is not here any more, if it ever was. */
  whyGone(chatId: string, name: string): string | undefined {
    const g = this.gone.get(chatId)
    return g?.get(safeName(name).toLowerCase()) ?? g?.get(String(name).toLowerCase())
  }

  /** The bytes from `start` to `end` inclusive, as a stream: what the loopback server sends. */
  readRange(id: string, start: number, end: number): Readable | null {
    const e = this.byId.get(id)
    if (!e) return null
    const last = Math.min(end, e.bytes - 1)
    if (e.file) return fs.createReadStream(e.file, { start, end: last })
    const blocks = e.blocks ?? []
    return Readable.from(
      (function* () {
        let offset = 0
        for (const block of blocks) {
          const from = offset
          const to = offset + block.length - 1
          offset += block.length
          if (to < start) continue
          if (from > last) break
          yield block.subarray(Math.max(0, start - from), Math.min(block.length, last - from + 1))
        }
      })(),
      { objectMode: false }
    )
  }

  /** How ffmpeg reaches a file: its path when it is in a folder, otherwise null (serve it). */
  pathOf(id: string): string | null {
    return this.byId.get(id)?.file ?? null
  }

  sizeOf(id: string): number | null {
    const e = this.byId.get(id)
    return e ? e.bytes : null
  }

  /** The whole of a text file, as text. */
  async readText(item: ScratchItem): Promise<string> {
    const e = this.byId.get(item.id)
    if (!e) throw new ScratchpadError(`${item.name} is no longer in the scratchpad.`)
    const bytes = e.file ? await fsp.readFile(e.file) : Buffer.concat(e.blocks ?? [])
    return bytes.toString('utf8')
  }

  /** Write a text file, replacing it, or adding to the end of it. */
  async writeText(chatId: string, name: string, text: string, append = false): Promise<ScratchItem> {
    const existing = this.get(chatId, name)
    const before = existing && append ? await this.readText(existing) : ''
    const next = before + text
    if (Buffer.byteLength(next) > MAX_TEXT_BYTES) {
      throw new ScratchpadError(`That would make ${existing?.name ?? safeName(name)} larger than ${MAX_TEXT_BYTES / 1024 / 1024} MB, the most a text file here can hold.`)
    }
    if (existing) await this.remove(chatId, existing.name, '')
    const bytes = Buffer.from(next, 'utf8')
    return this.add(chatId, existing?.name ?? name, Readable.from([bytes]), 'text', bytes.length)
  }

  /**
   * Delete a file. `reason` is kept, so the model is told why it is gone if it asks for it again;
   * an empty reason (a replacement, say) leaves no note.
   */
  async remove(chatId: string, name: string, reason: string): Promise<boolean> {
    const items = this.chats.get(chatId)
    const key = items?.has(safeName(name).toLowerCase()) ? safeName(name).toLowerCase() : String(name).toLowerCase()
    const e = items?.get(key)
    if (!items || !e) return false
    items.delete(key)
    this.byId.delete(e.id)
    if (!items.size) this.chats.delete(chatId)
    e.blocks = undefined
    if (e.file) await fsp.rm(e.file, { force: true }).catch(() => undefined)
    if (reason) {
      let g = this.gone.get(chatId)
      if (!g) {
        g = new Map()
        this.gone.set(chatId, g)
      }
      g.set(key, reason)
      // Notes are small, but a long conversation should not collect them without end.
      if (g.size > 200) g.delete(g.keys().next().value as string)
    }
    return true
  }

  /** Empty a conversation's scratchpad, as when the conversation is deleted. */
  async clearChat(chatId: string, reason = ''): Promise<number> {
    const names = [...(this.chats.get(chatId)?.values() ?? [])].map((e) => e.name)
    for (const name of names) await this.remove(chatId, name, reason)
    if (!reason) this.gone.delete(chatId)
    const root = this.ownedRoot()
    if (root) await fsp.rm(path.join(root, folderFor(chatId)), { recursive: true, force: true }).catch(() => undefined)
    return names.length
  }

  /** Delete every file the model has not read or written for `idleMs`. */
  async sweep(idleMs: number): Promise<ScratchItem[]> {
    const cutoff = this.now() - idleMs
    const stale = [...this.byId.values()].filter((e) => e.usedAt <= cutoff).map(publicItem)
    for (const item of stale) {
      const minutes = Math.round(idleMs / 60_000)
      await this.remove(item.chatId, item.name, `it was cleared automatically after ${minutes} minute${minutes === 1 ? '' : 's'} without being used`)
    }
    return stale
  }

  /** Everything, everywhere: memory released, and the app's own folder emptied. */
  async clearAll(): Promise<void> {
    for (const chatId of [...this.chats.keys()]) await this.clearChat(chatId)
    this.gone.clear()
    await clearOwnedFolder(this.opts.folder())
  }
}

/**
 * Empty the app's own folder inside a chosen location: never the location itself, and never
 * anything outside it, so pointing the setting at a folder of the user's own cannot cost them a file.
 */
export async function clearOwnedFolder(folder: string | null): Promise<void> {
  if (!folder) return
  await fsp.rm(path.join(folder, OWNED_FOLDER), { recursive: true, force: true }).catch(() => undefined)
}

function publicItem(e: Entry): ScratchItem {
  return { id: e.id, chatId: e.chatId, name: e.name, kind: e.kind, bytes: e.bytes, createdAt: e.createdAt, usedAt: e.usedAt }
}

function gb(n: number): string {
  return n >= 1024 ** 3 ? `${(n / 1024 ** 3).toFixed(1)} GB` : `${Math.round(n / 1024 ** 2)} MB`
}
