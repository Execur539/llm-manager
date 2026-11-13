/**
 * Which conversations and collections are protected, and the keys that open them.
 *
 * A protected conversation has its own random key. Without a password that key is kept wrapped by
 * the Windows account (Electron's safeStorage, which is DPAPI), so the app opens it by itself and
 * nothing that reads the files directly can. With a password it is wrapped only under the password
 * and the recovery key, and is in memory from the moment it is unlocked until it is locked again or
 * the app closes.
 *
 * Everything that stores or reads a protected value goes through a Box from here, so the rest of
 * the app reads and writes plain text as before.
 */

import path from 'node:path'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import { safeStorage } from 'electron'
import { all, get } from '../storage/db'
import { ATTACHMENTS_DIR } from '../storage/paths'
import {
  isSealedBlob,
  isSealedFileHead,
  isSealedText,
  openBlob,
  openBuffer,
  openText,
  sealBlob,
  sealBuffer,
  sealText,
  type Wrap
} from './cipher'

export type OwnerKind = 'chat' | 'collection'

export interface Owner {
  kind: OwnerKind
  id: string
}

/** What is stored in a protected row's `protection` column. */
export interface Protection {
  v: 1
  /** The key, wrapped by the Windows account: present when no password is set. */
  os?: string
  /** The key under the password and under the recovery key: present when a password is set. */
  pw?: Wrap
  rk?: Wrap
}

export interface ProtectStatus {
  protected: boolean
  locked: boolean
  password: boolean
}

export class LockedError extends Error {
  constructor(kind: OwnerKind = 'chat') {
    super(kind === 'chat' ? 'This conversation is locked. Unlock it to continue.' : 'This collection is locked. Unlock it to use it.')
    this.name = 'LockedError'
  }
}

const tableOf = (kind: OwnerKind): string => (kind === 'chat' ? 'chats' : 'collections')
const ref = (o: Owner): string => `${o.kind}:${o.id}`

const records = new Map<string, Protection | null>()
const keys = new Map<string, Buffer>()

function parse(raw: string | null | undefined): Protection | null {
  if (!raw) return null
  try {
    const p = JSON.parse(raw) as Protection
    return p && p.v === 1 ? p : null
  } catch {
    return null
  }
}

/** The row's protection, read once and then remembered. */
export function recordOf(owner: Owner): Protection | null {
  const k = ref(owner)
  if (records.has(k)) return records.get(k) ?? null
  const row = get<{ protection: string | null }>(`SELECT protection FROM ${tableOf(owner.kind)} WHERE id = ?`, owner.id)
  const p = parse(row?.protection)
  records.set(k, p)
  return p
}

/** What a listing already read, so a list of a hundred conversations is not a hundred more queries. */
export function primeRecord(owner: Owner, raw: string | null | undefined): void {
  records.set(ref(owner), parse(raw))
}

/** After the protection column was written. */
export function setRecord(owner: Owner, p: Protection | null): void {
  records.set(ref(owner), p)
}

export function forgetOwner(owner: Owner): void {
  records.delete(ref(owner))
  keys.delete(ref(owner))
}

export function osKeyStoreAvailable(): boolean {
  try {
    return safeStorage.isEncryptionAvailable()
  } catch {
    return false
  }
}

export function osWrap(dataKey: Buffer): string {
  if (!osKeyStoreAvailable()) {
    throw new Error('This computer has no key store to protect it with. Set a password instead.')
  }
  return safeStorage.encryptString(dataKey.toString('base64')).toString('base64')
}

function osUnwrap(wrapped: string): Buffer | null {
  try {
    return Buffer.from(safeStorage.decryptString(Buffer.from(wrapped, 'base64')), 'base64')
  } catch {
    return null
  }
}

/** The key that opens it, or null when it is locked (or not protected at all). */
export function keyOf(owner: Owner): Buffer | null {
  const k = ref(owner)
  const held = keys.get(k)
  if (held) return held
  const p = recordOf(owner)
  if (!p?.os) return null
  const key = osUnwrap(p.os)
  if (key) keys.set(k, key)
  return key
}

export function holdKey(owner: Owner, key: Buffer): void {
  keys.set(ref(owner), key)
}

export function dropKey(owner: Owner): void {
  keys.delete(ref(owner))
}

export function statusOf(owner: Owner): ProtectStatus {
  const p = recordOf(owner)
  if (!p) return { protected: false, locked: false, password: false }
  return { protected: true, locked: !keyOf(owner), password: !!p.pw }
}

export function isLocked(owner: Owner): boolean {
  return !!recordOf(owner) && !keyOf(owner)
}

/** Every protected row of a kind, for the few things that have to look across all of them. */
export function protectedIds(kind: OwnerKind): string[] {
  const rows = all<{ id: string; protection: string }>(
    `SELECT id, protection FROM ${tableOf(kind)} WHERE protection IS NOT NULL`
  )
  for (const r of rows) primeRecord({ kind, id: r.id }, r.protection)
  return rows.map((r) => r.id)
}

// ------------------------------------------------------------------ boxes

/** How one owner's values are stored: as they are, or encrypted under its key. */
export interface Box {
  readonly on: boolean
  readonly locked: boolean
  seal(v: string): string
  open(v: string): string
  sealOrNull(v: string | null | undefined): string | null
  openOrNull(v: string | null | undefined): string | null
  sealBlob(b: Uint8Array): Uint8Array
  openBlob(b: Uint8Array): Uint8Array
}

const PLAIN: Box = {
  on: false,
  locked: false,
  seal: (v) => v,
  open: (v) => v,
  sealOrNull: (v) => v ?? null,
  openOrNull: (v) => v ?? null,
  sealBlob: (b) => b,
  openBlob: (b) => b
}

export function keyBox(key: Buffer): Box {
  return {
    on: true,
    locked: false,
    seal: (v) => sealText(key, v),
    open: (v) => openText(key, v),
    sealOrNull: (v) => (v == null ? null : sealText(key, v)),
    openOrNull: (v) => (v == null ? null : openText(key, v)),
    sealBlob: (b) => sealBlob(key, b),
    openBlob: (b) => openBlob(key, b)
  }
}

function lockedBox(kind: OwnerKind): Box {
  const refuse = (): never => {
    throw new LockedError(kind)
  }
  const read = (v: string): string => (isSealedText(v) ? refuse() : v)
  return {
    on: true,
    locked: true,
    seal: refuse,
    open: read,
    sealOrNull: (v) => (v == null ? null : refuse()),
    openOrNull: (v) => (v == null ? null : read(v)),
    sealBlob: refuse,
    openBlob: (b) => (isSealedBlob(b) ? refuse() : b)
  }
}

export function plainBox(): Box {
  return PLAIN
}

export function boxFor(owner: Owner): Box {
  if (!recordOf(owner)) return PLAIN
  const key = keyOf(owner)
  return key ? keyBox(key) : lockedBox(owner.kind)
}

export const chatBox = (id: string): Box => boxFor({ kind: 'chat', id })
export const collectionBox = (id: string): Box => boxFor({ kind: 'collection', id })

// ------------------------------------------------------------------ files

/** The conversation a kept file belongs to: its folder under the attachments directory is named for it. */
export function chatOfFile(file: string): string | null {
  const rel = path.relative(ATTACHMENTS_DIR, path.resolve(file))
  if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) return null
  const first = rel.split(path.sep)[0]
  return first && first !== rel ? first : null
}

/** A file's bytes, decrypted when it is one of a protected conversation's kept files. */
export async function readOpen(file: string): Promise<Buffer> {
  const bytes = await fsp.readFile(file)
  if (!isSealedFileHead(bytes)) return bytes
  const chat = chatOfFile(file)
  const key = chat ? keyOf({ kind: 'chat', id: chat }) : null
  if (!key) throw new LockedError('chat')
  return openBuffer(key, bytes)
}

/** The same, for the media route, which answers synchronously. */
export function readOpenSync(file: string): Buffer {
  const bytes = fs.readFileSync(file)
  if (!isSealedFileHead(bytes)) return bytes
  const chat = chatOfFile(file)
  const key = chat ? keyOf({ kind: 'chat', id: chat }) : null
  if (!key) throw new LockedError('chat')
  return openBuffer(key, bytes)
}

/** Whether a file begins as an encrypted file does, without reading the rest of it. */
export function isSealedSync(file: string): boolean {
  let fd: number | null = null
  try {
    fd = fs.openSync(file, 'r')
    const head = Buffer.alloc(8)
    return fs.readSync(fd, head, 0, 8, 0) === 8 && isSealedFileHead(head)
  } catch {
    return false
  } finally {
    if (fd !== null) fs.closeSync(fd)
  }
}

/** Bytes for a conversation's kept file, encrypted when the conversation is protected. */
export function bytesFor(chatId: string, bytes: Buffer): Buffer {
  const box = chatBox(chatId)
  if (!box.on) return bytes
  const key = keyOf({ kind: 'chat', id: chatId })
  if (!key) throw new LockedError('chat')
  return sealBuffer(key, bytes)
}
