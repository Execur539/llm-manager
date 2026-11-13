/**
 * The encryption protected conversations and collections are stored with.
 *
 * Plain Node, no Electron, so it can be checked on its own (scripts/protect-check.mjs).
 *
 * - Text is AES-256-GCM, stored as a marker then base64 of nonce, tag and ciphertext. The marker
 *   begins with an invisible character nobody types, so a stored value is never mistaken for one.
 * - Files are the same cipher in chunks of 1 MiB, each with its own nonce (a random prefix and the
 *   chunk's number) and its own tag, the last one marked as the last. A file of any size can be
 *   streamed through it, and a file that was cut short, reordered or altered fails to open rather
 *   than opening wrong.
 * - Keys are 32 random bytes per conversation. A password or recovery key never encrypts anything
 *   itself: it is stretched with scrypt into a key that wraps the conversation's key, so changing a
 *   password rewraps 32 bytes instead of rewriting the conversation.
 */

import crypto from 'node:crypto'
import fsp from 'node:fs/promises'

export const TEXT_MARK = '⁣lk1:'
const BLOB_MAGIC = Buffer.from('LKB1', 'ascii')
const FILE_MAGIC = Buffer.from('LLMMLK01', 'ascii')
const HEADER = FILE_MAGIC.length + 8
const CHUNK = 1024 * 1024
const NONCE = 12
const TAG = 16

export function newKey(): Buffer {
  return crypto.randomBytes(32)
}

function seal(key: Buffer, plain: Buffer): Buffer {
  const iv = crypto.randomBytes(NONCE)
  const c = crypto.createCipheriv('aes-256-gcm', key, iv)
  const body = Buffer.concat([c.update(plain), c.final()])
  return Buffer.concat([iv, c.getAuthTag(), body])
}

function open(key: Buffer, sealed: Buffer): Buffer {
  if (sealed.length < NONCE + TAG) throw new Error('Not encrypted data')
  const d = crypto.createDecipheriv('aes-256-gcm', key, sealed.subarray(0, NONCE))
  d.setAuthTag(sealed.subarray(NONCE, NONCE + TAG))
  return Buffer.concat([d.update(sealed.subarray(NONCE + TAG)), d.final()])
}

// ------------------------------------------------------------------ text and small values

export function isSealedText(v: unknown): v is string {
  return typeof v === 'string' && v.startsWith(TEXT_MARK)
}

export function sealText(key: Buffer, text: string): string {
  return TEXT_MARK + seal(key, Buffer.from(text, 'utf8')).toString('base64')
}

/** A value that is not encrypted comes back as it is. */
export function openText(key: Buffer, value: string): string {
  if (!isSealedText(value)) return value
  return open(key, Buffer.from(value.slice(TEXT_MARK.length), 'base64')).toString('utf8')
}

export function isSealedBlob(b: Uint8Array): boolean {
  return b.length >= BLOB_MAGIC.length + NONCE + TAG && Buffer.from(b.buffer, b.byteOffset, BLOB_MAGIC.length).equals(BLOB_MAGIC)
}

export function sealBlob(key: Buffer, plain: Uint8Array): Buffer {
  return Buffer.concat([BLOB_MAGIC, seal(key, Buffer.from(plain.buffer, plain.byteOffset, plain.byteLength))])
}

export function openBlob(key: Buffer, b: Uint8Array): Buffer {
  const buf = Buffer.from(b.buffer, b.byteOffset, b.byteLength)
  return isSealedBlob(buf) ? open(key, buf.subarray(BLOB_MAGIC.length)) : Buffer.from(buf)
}

// ------------------------------------------------------------------ files

function chunkIv(prefix: Buffer, index: number): Buffer {
  const iv = Buffer.alloc(NONCE)
  prefix.copy(iv, 0)
  iv.writeUInt32BE(index, 8)
  return iv
}

function sealChunk(key: Buffer, prefix: Buffer, index: number, last: boolean, plain: Buffer): Buffer {
  const c = crypto.createCipheriv('aes-256-gcm', key, chunkIv(prefix, index))
  c.setAAD(Buffer.from([last ? 1 : 0]))
  const body = Buffer.concat([c.update(plain), c.final()])
  return Buffer.concat([c.getAuthTag(), body])
}

function openChunk(key: Buffer, prefix: Buffer, index: number, last: boolean, sealed: Buffer): Buffer {
  if (sealed.length < TAG) throw new Error('The encrypted file is damaged')
  const d = crypto.createDecipheriv('aes-256-gcm', key, chunkIv(prefix, index))
  d.setAAD(Buffer.from([last ? 1 : 0]))
  d.setAuthTag(sealed.subarray(0, TAG))
  return Buffer.concat([d.update(sealed.subarray(TAG)), d.final()])
}

const chunksFor = (plainBytes: number): number => Math.max(1, Math.ceil(plainBytes / CHUNK))
const sealedChunksFor = (sealedBytes: number): number => Math.max(1, Math.ceil((sealedBytes - HEADER) / (CHUNK + TAG)))

export function isSealedFileHead(head: Uint8Array): boolean {
  return head.length >= FILE_MAGIC.length && Buffer.from(head.buffer, head.byteOffset, FILE_MAGIC.length).equals(FILE_MAGIC)
}

export async function isSealedFile(file: string): Promise<boolean> {
  const fh = await fsp.open(file, 'r').catch(() => null)
  if (!fh) return false
  try {
    const head = Buffer.alloc(FILE_MAGIC.length)
    const { bytesRead } = await fh.read(head, 0, head.length, 0)
    return bytesRead === head.length && isSealedFileHead(head)
  } finally {
    await fh.close()
  }
}

/** A whole file's worth of bytes, encrypted in the file format. */
export function sealBuffer(key: Buffer, plain: Buffer): Buffer {
  const prefix = crypto.randomBytes(8)
  const count = chunksFor(plain.length)
  const parts: Buffer[] = [FILE_MAGIC, prefix]
  for (let i = 0; i < count; i++) {
    parts.push(sealChunk(key, prefix, i, i === count - 1, plain.subarray(i * CHUNK, (i + 1) * CHUNK)))
  }
  return Buffer.concat(parts)
}

export function openBuffer(key: Buffer, sealed: Buffer): Buffer {
  if (!isSealedFileHead(sealed)) throw new Error('Not an encrypted file')
  const prefix = sealed.subarray(FILE_MAGIC.length, HEADER)
  const count = sealedChunksFor(sealed.length)
  const parts: Buffer[] = []
  for (let i = 0; i < count; i++) {
    const at = HEADER + i * (CHUNK + TAG)
    parts.push(openChunk(key, prefix, i, i === count - 1, sealed.subarray(at, at + CHUNK + TAG)))
  }
  return Buffer.concat(parts)
}

/** Encrypt `src` into `dst`, a chunk at a time, so a file of several gigabytes never sits in memory. */
export async function sealFile(key: Buffer, src: string, dst: string): Promise<void> {
  const input = await fsp.open(src, 'r')
  const output = await fsp.open(dst, 'w')
  try {
    const prefix = crypto.randomBytes(8)
    await output.write(Buffer.concat([FILE_MAGIC, prefix]))
    const count = chunksFor((await input.stat()).size)
    const buf = Buffer.allocUnsafe(CHUNK)
    for (let i = 0; i < count; i++) {
      const { bytesRead } = await input.read(buf, 0, CHUNK, i * CHUNK)
      await output.write(sealChunk(key, prefix, i, i === count - 1, buf.subarray(0, bytesRead)))
    }
  } finally {
    await input.close()
    await output.close()
  }
}

/** Decrypt `src` into `dst`, a chunk at a time. */
export async function openFile(key: Buffer, src: string, dst: string): Promise<void> {
  const input = await fsp.open(src, 'r')
  const output = await fsp.open(dst, 'w')
  try {
    const size = (await input.stat()).size
    const head = Buffer.alloc(HEADER)
    await input.read(head, 0, HEADER, 0)
    if (!isSealedFileHead(head)) throw new Error('Not an encrypted file')
    const prefix = head.subarray(FILE_MAGIC.length)
    const count = sealedChunksFor(size)
    const buf = Buffer.allocUnsafe(CHUNK + TAG)
    for (let i = 0; i < count; i++) {
      const { bytesRead } = await input.read(buf, 0, CHUNK + TAG, HEADER + i * (CHUNK + TAG))
      await output.write(openChunk(key, prefix, i, i === count - 1, buf.subarray(0, bytesRead)))
    }
  } finally {
    await input.close()
    await output.close()
  }
}

/** Re-encrypt `src` from one key to another into `dst`, a chunk at a time, never writing it out plain. */
export async function resealFile(from: Buffer, to: Buffer, src: string, dst: string): Promise<void> {
  const input = await fsp.open(src, 'r')
  const output = await fsp.open(dst, 'w')
  try {
    const size = (await input.stat()).size
    const head = Buffer.alloc(HEADER)
    await input.read(head, 0, HEADER, 0)
    if (!isSealedFileHead(head)) throw new Error('Not an encrypted file')
    const oldPrefix = head.subarray(FILE_MAGIC.length)
    const newPrefix = crypto.randomBytes(8)
    await output.write(Buffer.concat([FILE_MAGIC, newPrefix]))
    const count = sealedChunksFor(size)
    const buf = Buffer.allocUnsafe(CHUNK + TAG)
    for (let i = 0; i < count; i++) {
      const last = i === count - 1
      const { bytesRead } = await input.read(buf, 0, CHUNK + TAG, HEADER + i * (CHUNK + TAG))
      await output.write(sealChunk(to, newPrefix, i, last, openChunk(from, oldPrefix, i, last, buf.subarray(0, bytesRead))))
    }
  } finally {
    await input.close()
    await output.close()
  }
}

// ------------------------------------------------------------------ passwords and recovery keys

/** A conversation key wrapped under a secret, with what it takes to unwrap it again. */
export interface Wrap {
  salt: string
  n: number
  r: number
  p: number
  key: string
}

/** About a third of a second and 128 MiB per guess: slow to guess, quick enough to type. */
export const PASSWORD_COST = 2 ** 17
/** A recovery key is 125 random bits; stretching it hard adds nothing. */
export const RECOVERY_COST = 2 ** 14

function stretch(secret: string, salt: Buffer, n: number, r: number, p: number): Promise<Buffer> {
  return new Promise((resolve, reject) =>
    crypto.scrypt(secret.normalize('NFC'), salt, 32, { N: n, r, p, maxmem: 256 * 1024 * 1024 }, (err, key) =>
      err ? reject(err) : resolve(key)
    )
  )
}

export async function wrapKey(secret: string, dataKey: Buffer, cost: number): Promise<Wrap> {
  const salt = crypto.randomBytes(16)
  const kek = await stretch(secret, salt, cost, 8, 1)
  return { salt: salt.toString('base64'), n: cost, r: 8, p: 1, key: seal(kek, dataKey).toString('base64') }
}

/** The conversation key, or null when the secret is not the one it was wrapped under. */
export async function unwrapKey(secret: string, w: Wrap): Promise<Buffer | null> {
  const kek = await stretch(secret, Buffer.from(w.salt, 'base64'), w.n, w.r, w.p)
  try {
    return open(kek, Buffer.from(w.key, 'base64'))
  } catch {
    return null
  }
}

/** Crockford's alphabet: no I, L, O or U, so nothing in it reads as something else. */
const ALPHABET = '0123456789ABCDEFGHJKMNPQRSTVWXYZ'

/** Twenty-five characters in five groups, like ABCDE-FGHJK-MNPQR-STVWX-YZ012. */
export function newRecoveryKey(): string {
  let out = ''
  for (let i = 0; i < 25; i++) out += ALPHABET[crypto.randomInt(32)]
  return out.match(/.{5}/g)!.join('-')
}

/** What was typed, as a recovery key: any case, any separators, O for 0 and I or L for 1. Null when it cannot be one. */
export function normaliseRecoveryKey(typed: string): string | null {
  const s = typed.toUpperCase().replace(/[^0-9A-Z]/g, '').replace(/O/g, '0').replace(/[IL]/g, '1')
  if (s.length !== 25 || [...s].some((c) => !ALPHABET.includes(c))) return null
  return s.match(/.{5}/g)!.join('-')
}
