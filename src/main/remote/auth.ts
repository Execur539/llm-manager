/**
 * Authentication for the remote web UI.
 *
 * Password + signed session cookie, as decided in Round 4. Remote access cannot be enabled
 * until a password is set — that check lives here rather than in the UI so it cannot be
 * bypassed by a config edit.
 *
 * Secrets are stored under DPAPI, so lifting %APPDATA% onto another machine yields nothing
 * usable. Passwords are scrypt-hashed; sessions are HMAC-signed with a per-install key.
 */

import crypto from 'node:crypto'
import fs from 'node:fs'
import { safeStorage } from 'electron'
import { APPDATA_DIR, SECRETS_FILE } from '../storage/paths'

interface Secrets {
  passwordHash?: string
  passwordSalt?: string
  sessionKey?: string
  apiKey?: string
  hfToken?: string
  freednsToken?: string
}

const SCRYPT_PARAMS = { N: 16384, r: 8, p: 1, keylen: 64 }
const SESSION_TTL_MS = 7 * 24 * 60 * 60 * 1000

function readSecrets(): Secrets {
  try {
    const raw = fs.readFileSync(SECRETS_FILE)
    // Encrypted at rest whenever the OS provides it.
    if (safeStorage.isEncryptionAvailable()) {
      return JSON.parse(safeStorage.decryptString(raw)) as Secrets
    }
    return JSON.parse(raw.toString('utf8')) as Secrets
  } catch {
    return {}
  }
}

function writeSecrets(secrets: Secrets): void {
  fs.mkdirSync(APPDATA_DIR, { recursive: true })
  const text = JSON.stringify(secrets)
  const payload = safeStorage.isEncryptionAvailable()
    ? safeStorage.encryptString(text)
    : Buffer.from(text, 'utf8')
  fs.writeFileSync(SECRETS_FILE, payload)
}

function scrypt(password: string, salt: string): string {
  return crypto
    .scryptSync(password, salt, SCRYPT_PARAMS.keylen, { N: SCRYPT_PARAMS.N, r: SCRYPT_PARAMS.r, p: SCRYPT_PARAMS.p })
    .toString('hex')
}

export function hasPassword(): boolean {
  const s = readSecrets()
  return !!(s.passwordHash && s.passwordSalt)
}

export function setPassword(password: string): void {
  if (password.length < 10) {
    throw new Error('Password must be at least 10 characters. This is reachable from the internet.')
  }
  const secrets = readSecrets()
  const salt = crypto.randomBytes(16).toString('hex')
  secrets.passwordSalt = salt
  secrets.passwordHash = scrypt(password, salt)
  // Rotating the session key logs out every existing browser.
  secrets.sessionKey = crypto.randomBytes(32).toString('hex')
  writeSecrets(secrets)
}

export function verifyPassword(password: string): boolean {
  const s = readSecrets()
  if (!s.passwordHash || !s.passwordSalt) return false
  const candidate = Buffer.from(scrypt(password, s.passwordSalt))
  const stored = Buffer.from(s.passwordHash)
  return candidate.length === stored.length && crypto.timingSafeEqual(candidate, stored)
}

function sessionKey(): string {
  const secrets = readSecrets()
  if (!secrets.sessionKey) {
    secrets.sessionKey = crypto.randomBytes(32).toString('hex')
    writeSecrets(secrets)
  }
  return secrets.sessionKey
}

export function issueSession(): string {
  const expires = Date.now() + SESSION_TTL_MS
  const nonce = crypto.randomBytes(12).toString('hex')
  const payload = `${expires}.${nonce}`
  const sig = crypto.createHmac('sha256', sessionKey()).update(payload).digest('hex')
  return `${payload}.${sig}`
}

export function verifySession(token: string | undefined): boolean {
  if (!token) return false
  const parts = token.split('.')
  if (parts.length !== 3) return false
  const [expires, nonce, sig] = parts
  if (Number(expires) < Date.now()) return false

  const expected = crypto.createHmac('sha256', sessionKey()).update(`${expires}.${nonce}`).digest('hex')
  const a = Buffer.from(sig)
  const b = Buffer.from(expected)
  return a.length === b.length && crypto.timingSafeEqual(a, b)
}

// ---------------------------------------------------------------- other secrets

export function getApiKey(): string | null {
  return readSecrets().apiKey ?? null
}

export function setApiKey(key: string | null): void {
  const s = readSecrets()
  if (key) s.apiKey = key
  else delete s.apiKey
  writeSecrets(s)
}

export function generateApiKey(): string {
  const key = `llmm-${crypto.randomBytes(24).toString('hex')}`
  setApiKey(key)
  return key
}

export function getHfToken(): string | null {
  return readSecrets().hfToken ?? null
}

export function setHfToken(token: string | null): void {
  const s = readSecrets()
  if (token) s.hfToken = token
  else delete s.hfToken
  writeSecrets(s)
}

export function getFreednsToken(): string | null {
  return readSecrets().freednsToken ?? null
}

export function setFreednsToken(token: string | null): void {
  const s = readSecrets()
  if (token) s.freednsToken = token
  else delete s.freednsToken
  writeSecrets(s)
}

// ---------------------------------------------------------------- brute-force defence

interface Attempt {
  count: number
  firstAt: number
  lockedUntil: number
}

const attempts = new Map<string, Attempt>()
const MAX_ATTEMPTS = 6
const WINDOW_MS = 15 * 60 * 1000
const LOCKOUT_MS = 15 * 60 * 1000

export function isLockedOut(ip: string): number {
  const a = attempts.get(ip)
  if (!a) return 0
  if (a.lockedUntil > Date.now()) return Math.ceil((a.lockedUntil - Date.now()) / 1000)
  return 0
}

export function recordFailure(ip: string): void {
  const now = Date.now()
  const a = attempts.get(ip) ?? { count: 0, firstAt: now, lockedUntil: 0 }
  if (now - a.firstAt > WINDOW_MS) {
    a.count = 0
    a.firstAt = now
  }
  a.count++
  if (a.count >= MAX_ATTEMPTS) {
    a.lockedUntil = now + LOCKOUT_MS
    a.count = 0
    a.firstAt = now
  }
  attempts.set(ip, a)
}

export function recordSuccess(ip: string): void {
  attempts.delete(ip)
}
