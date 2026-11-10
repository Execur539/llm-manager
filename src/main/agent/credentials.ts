/**
 * Saved logins: usernames and passwords the agent keeps between conversations.
 *
 * So that "sign in to the router and restart it" does not need the password typed out again each
 * time, and so the agent can sign in to things while nobody is watching.
 *
 * Passwords are encrypted with the operating system's own key store wherever it offers one — the
 * same arrangement the app's own secrets use. That is not about the model: it is so the database
 * file, which every read-tier tool can open, a backup can copy and a sync folder can upload, does
 * not have the passwords sitting in it as text.
 */

import { safeStorage } from 'electron'
import { all, get, run } from '../storage/db'
import type { ResolvedLogin } from './credential-refs'

export interface SavedLogin {
  name: string
  username: string
  url: string | null
  notes: string | null
  updatedAt: number
}

interface Row {
  name: string
  username: string | null
  secret: string
  url: string | null
  notes: string | null
  updated_at: number
}

const MAX_FIELD = 2000

function seal(password: string): string {
  if (safeStorage.isEncryptionAvailable()) {
    return `os:${safeStorage.encryptString(password).toString('base64')}`
  }
  return `plain:${Buffer.from(password, 'utf8').toString('base64')}`
}

function open(secret: string): string {
  const at = secret.indexOf(':')
  const kind = secret.slice(0, at)
  const body = Buffer.from(secret.slice(at + 1), 'base64')
  return kind === 'os' ? safeStorage.decryptString(body) : body.toString('utf8')
}

const toLogin = (r: Row): SavedLogin => ({
  name: r.name,
  username: r.username ?? '',
  url: r.url,
  notes: r.notes,
  updatedAt: r.updated_at
})

/** Every saved login, without its password. */
export function listLogins(): SavedLogin[] {
  return all<Row>('SELECT * FROM credentials ORDER BY name COLLATE NOCASE').map(toLogin)
}

/**
 * Save a login, replacing any saved under the same name.
 *
 * Names are matched without regard to case, so "GitHub" and "github" are one login rather than
 * two that the model then has to choose between.
 */
export function saveLogin(input: {
  name: string
  username?: string
  password: string
  url?: string | null
  notes?: string | null
}): SavedLogin {
  const name = input.name.trim().slice(0, 120)
  if (!name) throw new Error('A saved login needs a name.')
  if (/[:{}]/.test(name)) throw new Error('A name cannot contain a colon or braces.')
  if (!input.password) throw new Error('A saved login needs a password.')
  const now = Date.now()
  run(
    `INSERT INTO credentials (name, username, secret, url, notes, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(name) DO UPDATE SET
       username = excluded.username, secret = excluded.secret, url = excluded.url,
       notes = excluded.notes, updated_at = excluded.updated_at`,
    name,
    (input.username ?? '').slice(0, MAX_FIELD),
    seal(input.password.slice(0, MAX_FIELD)),
    input.url?.trim().slice(0, MAX_FIELD) || null,
    input.notes?.trim().slice(0, MAX_FIELD) || null,
    now,
    now
  )
  const row = get<Row>('SELECT * FROM credentials WHERE name = ? COLLATE NOCASE', name)
  if (!row) throw new Error('The login could not be saved.')
  return toLogin(row)
}

/** The username and password saved under a name, or null when there is none. */
export function resolveLogin(name: string): ResolvedLogin | null {
  const row = get<Row>('SELECT * FROM credentials WHERE name = ? COLLATE NOCASE', name.trim())
  if (!row) return null
  try {
    return { username: row.username ?? '', password: open(row.secret) }
  } catch {
    // Encrypted under a different Windows account, or copied from another machine.
    return null
  }
}

export function deleteLogin(name: string): boolean {
  const before = get<{ n: number }>('SELECT COUNT(*) AS n FROM credentials WHERE name = ? COLLATE NOCASE', name.trim())
  run('DELETE FROM credentials WHERE name = ? COLLATE NOCASE', name.trim())
  return (before?.n ?? 0) > 0
}
