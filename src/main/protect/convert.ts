/**
 * Turning protection on and off, and the password on and off.
 *
 * Turning it on encrypts everything the conversation (or collection) already has where it is
 * stored: the database rows in one transaction, and its kept files, which are written encrypted
 * under new names beside the old ones first and only replace them once the rows say so. Turning it
 * off is the same in reverse. A password changes none of that: it only changes what the
 * conversation's key is wrapped under.
 */

import path from 'node:path'
import crypto from 'node:crypto'
import fsp from 'node:fs/promises'
import { all, get, getDb, run, transaction } from '../storage/db'
import { ATTACHMENTS_DIR } from '../storage/paths'
import {
  PASSWORD_COST,
  RECOVERY_COST,
  isSealedFile,
  newKey,
  newRecoveryKey,
  normaliseRecoveryKey,
  openFile,
  resealFile,
  sealFile,
  unwrapKey,
  wrapKey
} from './cipher'
import {
  LockedError,
  dropKey,
  holdKey,
  keyBox,
  keyOf,
  osWrap,
  plainBox,
  recordOf,
  setRecord,
  type Box,
  type Owner,
  type Protection
} from './keys'

const tableOf = (o: Owner): string => (o.kind === 'chat' ? 'chats' : 'collections')
const segment = (s: string): string => s.replace(/[^A-Za-z0-9_-]/g, '_')

function recode(table: string, where: string, args: unknown[], cols: string[], from: Box, to: Box): void {
  const rows = all<Record<string, unknown>>(`SELECT rowid AS _r, ${cols.join(', ')} FROM ${table} WHERE ${where}`, ...args)
  for (const row of rows) {
    const values = cols.map((c) => {
      const v = row[c]
      return typeof v === 'string' ? to.seal(from.open(v)) : v
    })
    run(`UPDATE ${table} SET ${cols.map((c) => `${c} = ?`).join(', ')} WHERE rowid = ?`, ...values, row._r)
  }
}

function recodeDocuments(where: string, args: unknown[], from: Box, to: Box): void {
  recode('documents', where.replace(/\bd\./g, ''), args, ['filename', 'path'], from, to)
  for (const r of all<{ r: number; text: string }>(
    `SELECT c.rowid AS r, c.text FROM chunks c JOIN documents d ON d.id = c.document_id WHERE ${where}`,
    ...args
  )) {
    run('UPDATE chunks SET text = ? WHERE rowid = ?', to.seal(from.open(r.text)), r.r)
  }
  for (const r of all<{ r: number; vector: Uint8Array }>(
    `SELECT e.rowid AS r, e.vector FROM embeddings e JOIN chunks c ON c.id = e.chunk_id JOIN documents d ON d.id = c.document_id WHERE ${where}`,
    ...args
  )) {
    run('UPDATE embeddings SET vector = ? WHERE rowid = ?', to.sealBlob(from.openBlob(r.vector)), r.r)
  }
}

function parseJson(raw: string | null): unknown {
  if (!raw) return null
  try {
    return JSON.parse(raw)
  } catch {
    return null
  }
}

function recodeChat(chatId: string, from: Box, to: Box, files: Map<string, string>): void {
  const moved = (p: unknown): unknown => (typeof p === 'string' ? (files.get(path.resolve(p)) ?? p) : p)

  recode('chats', 'id = ?', [chatId], ['title', 'summary', 'prompt_state', 'system_prompt'], from, to)
  recode('messages', 'chat_id = ?', [chatId], ['content', 'reasoning', 'plan', 'tool_calls', 'tool_result', 'follow_up'], from, to)
  recode('tasks', 'chat_id = ?', [chatId], ['text'], from, to)

  for (const a of all<{ r: number; path: string; meta: string | null }>(
    'SELECT a.rowid AS r, a.path, a.meta FROM attachments a JOIN messages m ON m.id = a.message_id WHERE m.chat_id = ?',
    chatId
  )) {
    const meta = (parseJson(from.openOrNull(a.meta)) ?? {}) as Record<string, unknown>
    const file = moved(a.path) as string
    // The kept copy's new name says nothing; the name the user saw goes with the rest of the record.
    if (file !== a.path && typeof meta.name !== 'string') meta.name = path.basename(a.path)
    if (meta.optimised) meta.optimised = moved(meta.optimised)
    if (Array.isArray(meta.stills)) meta.stills = meta.stills.map(moved)
    const text = Object.keys(meta).length ? JSON.stringify(meta) : null
    run('UPDATE attachments SET path = ?, meta = ? WHERE rowid = ?', file, to.sealOrNull(text), a.r)
  }

  for (const c of all<{ r: number; content: string }>('SELECT rowid AS r, content FROM message_content WHERE chat_id = ?', chatId)) {
    let text = from.open(c.content)
    const parts = parseJson(text)
    if (Array.isArray(parts) && files.size) {
      text = JSON.stringify(parts.map((p) => (p && typeof p === 'object' && 'file' in p ? { ...p, file: moved(p.file) } : p)))
    }
    run('UPDATE message_content SET content = ? WHERE rowid = ?', to.seal(text), c.r)
  }

  recodeDocuments('d.chat_id = ? AND d.collection_id IS NULL', [chatId], from, to)
}

function recodeOwner(owner: Owner, from: Box, to: Box, files: Map<string, string>): void {
  if (owner.kind === 'chat') {
    recodeChat(owner.id, from, to, files)
  } else {
    recode('collections', 'id = ?', [owner.id], ['name'], from, to)
    recodeDocuments('d.collection_id = ?', [owner.id], from, to)
  }
}

async function filesUnder(dir: string): Promise<string[]> {
  const out: string[] = []
  for (const entry of await fsp.readdir(dir, { withFileTypes: true }).catch(() => [])) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) out.push(...(await filesUnder(full)))
    else if (entry.isFile()) out.push(full)
  }
  return out
}

/**
 * Write every kept file of a conversation again, encrypted (`to`) or decrypted (`from`), under a
 * new name beside the old one. Returns old path to new; the old files are still there.
 */
async function recodeFiles(chatId: string, from: Buffer | null, to: Buffer | null): Promise<Map<string, string>> {
  const map = new Map<string, string>()
  try {
    for (const file of await filesUnder(path.join(ATTACHMENTS_DIR, segment(chatId)))) {
      const sealed = await isSealedFile(file)
      // Re-keying touches only encrypted files; encrypting only plain ones; decrypting only encrypted ones.
      if (from && to ? !sealed : to ? sealed : !sealed) continue
      const next = path.join(path.dirname(file), crypto.randomBytes(8).toString('hex') + path.extname(file).toLowerCase())
      if (from && to) await resealFile(from, to, file, next)
      else if (to) await sealFile(to, file, next)
      else if (from) await openFile(from, file, next)
      map.set(path.resolve(file), next)
    }
  } catch (err) {
    await removeFiles([...map.values()])
    throw err
  }
  return map
}

async function removeFiles(files: string[]): Promise<void> {
  for (const f of files) await fsp.rm(f, { force: true }).catch(() => undefined)
}

function mustExist(owner: Owner): void {
  if (!get(`SELECT 1 AS x FROM ${tableOf(owner)} WHERE id = ?`, owner.id)) {
    throw new Error(owner.kind === 'chat' ? 'No such conversation' : 'No such collection')
  }
}

async function passwordRecord(password: string, dataKey: Buffer): Promise<{ record: Protection; recoveryKey: string }> {
  if (password.length < 8) throw new Error('Use at least 8 characters for the password.')
  const recoveryKey = newRecoveryKey()
  return {
    record: { v: 1, pw: await wrapKey(password, dataKey, PASSWORD_COST), rk: await wrapKey(recoveryKey, dataKey, RECOVERY_COST) },
    recoveryKey
  }
}

/** Encrypt everything it has, under a new key: wrapped by Windows, or by a password when one is given. */
export async function protect(owner: Owner, password?: string): Promise<{ recoveryKey?: string }> {
  mustExist(owner)
  if (recordOf(owner)) return {}
  const dataKey = newKey()
  const { record, recoveryKey } = password
    ? await passwordRecord(password, dataKey)
    : { record: { v: 1, os: osWrap(dataKey) } as Protection, recoveryKey: undefined }
  const files = owner.kind === 'chat' ? await recodeFiles(owner.id, null, dataKey) : new Map<string, string>()
  try {
    transaction(() => {
      run(`UPDATE ${tableOf(owner)} SET protection = ? WHERE id = ?`, JSON.stringify(record), owner.id)
      recodeOwner(owner, plainBox(), keyBox(dataKey), files)
    })
  } catch (err) {
    await removeFiles([...files.values()])
    throw err
  }
  setRecord(owner, record)
  holdKey(owner, dataKey)
  await removeFiles([...files.keys()])
  dropOldCopies()
  return { recoveryKey }
}

/**
 * Rewriting a row leaves its old version behind: in the write-ahead log until the next checkpoint,
 * and in the free space of the database file after it. Protecting something would then leave its
 * plain text readable in both. The file is rebuilt with only live rows in it and the log emptied,
 * so what was just encrypted is not still there unencrypted. Best effort: a failure here leaves
 * the protection itself in place.
 */
function dropOldCopies(): void {
  try {
    getDb().exec('VACUUM')
    getDb().exec('PRAGMA wal_checkpoint(TRUNCATE)')
  } catch {
    /* tried again by the next protection */
  }
}

/** Decrypt everything it has where it is stored, and forget its key. Only while it is unlocked. */
export async function unprotect(owner: Owner): Promise<void> {
  if (!recordOf(owner)) return
  const dataKey = keyOf(owner)
  if (!dataKey) throw new LockedError(owner.kind)
  const files = owner.kind === 'chat' ? await recodeFiles(owner.id, dataKey, null) : new Map<string, string>()
  try {
    transaction(() => {
      run(`UPDATE ${tableOf(owner)} SET protection = NULL WHERE id = ?`, owner.id)
      recodeOwner(owner, keyBox(dataKey), plainBox(), files)
    })
  } catch (err) {
    await removeFiles([...files.values()])
    throw err
  }
  setRecord(owner, null)
  dropKey(owner)
  await removeFiles([...files.keys()])
}

/** Put it under a password (or a new one), with a new recovery key, returned once and kept nowhere. */
export async function setPassword(owner: Owner, password: string): Promise<{ recoveryKey: string }> {
  // Not protected yet: protected under the password straight away, never by Windows on the way.
  if (!recordOf(owner)) {
    const { recoveryKey } = await protect(owner, password)
    return { recoveryKey: recoveryKey! }
  }
  const dataKey = keyOf(owner)
  if (!dataKey) throw new LockedError(owner.kind)
  if (recordOf(owner)?.os) {
    /*
     * Its first password. Until now its key could be had by anything running as this Windows
     * account, so the key changes with the password: everything is encrypted again under a new
     * one that has only ever been wrapped by the password and the recovery key, and the old
     * record (the key Windows could unwrap) is cleared out of the file.
     */
    const fresh = newKey()
    const { record, recoveryKey } = await passwordRecord(password, fresh)
    const files = owner.kind === 'chat' ? await recodeFiles(owner.id, dataKey, fresh) : new Map<string, string>()
    try {
      transaction(() => {
        run(`UPDATE ${tableOf(owner)} SET protection = ? WHERE id = ?`, JSON.stringify(record), owner.id)
        recodeOwner(owner, keyBox(dataKey), keyBox(fresh), files)
      })
    } catch (err) {
      await removeFiles([...files.values()])
      throw err
    }
    setRecord(owner, record)
    holdKey(owner, fresh)
    await removeFiles([...files.keys()])
    dropOldCopies()
    return { recoveryKey }
  }
  // A new password for one that had one: the same key, wrapped again.
  const { record, recoveryKey } = await passwordRecord(password, dataKey)
  run(`UPDATE ${tableOf(owner)} SET protection = ? WHERE id = ?`, JSON.stringify(record), owner.id)
  setRecord(owner, record)
  holdKey(owner, dataKey)
  dropOldCopies()
  return { recoveryKey }
}

/** Back to opening by itself in the app: the key wrapped by Windows again, password and recovery key gone. */
export function removePassword(owner: Owner): void {
  const dataKey = keyOf(owner)
  if (!recordOf(owner)) return
  if (!dataKey) throw new LockedError(owner.kind)
  const record: Protection = { v: 1, os: osWrap(dataKey) }
  run(`UPDATE ${tableOf(owner)} SET protection = ? WHERE id = ?`, JSON.stringify(record), owner.id)
  setRecord(owner, record)
}

/** Open it with its password or its recovery key. False when it is neither. */
export async function unlock(owner: Owner, secret: string): Promise<boolean> {
  const record = recordOf(owner)
  if (!record || keyOf(owner)) return true
  if (record.pw) {
    const key = await unwrapKey(secret, record.pw)
    if (key) {
      holdKey(owner, key)
      return true
    }
  }
  const recovery = normaliseRecoveryKey(secret)
  if (recovery && record.rk) {
    const key = await unwrapKey(recovery, record.rk)
    if (key) {
      holdKey(owner, key)
      return true
    }
  }
  return false
}

/** Forget the key until it is unlocked again. Only a conversation with a password can be locked. */
export function lock(owner: Owner): boolean {
  if (!recordOf(owner)?.pw) return false
  dropKey(owner)
  return true
}
