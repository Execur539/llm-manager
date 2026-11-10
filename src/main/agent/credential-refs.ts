/**
 * References to saved logins inside tool arguments.
 *
 * A saved password is used by writing `{{credential:NAME}}` where it belongs in a tool call —
 * the text typed into a sign-in form, part of a command. The reference is swapped for the real
 * value at the moment the tool runs and at no other point, so the password is never in the
 * model's context, the transcript or the approval prompt. `{{credential:NAME:username}}` gives
 * the username the same way.
 *
 * Plain functions of their inputs, kept apart from storage so they can be tested without a
 * database or the operating system's encryption.
 */

import type { ToolCall } from '@shared/types'

/** `{{credential:NAME}}` or `{{credential:NAME:password}}` or `{{credential:NAME:username}}`. */
const REFERENCE = /\{\{\s*credential\s*:\s*([^:{}]+?)\s*(?::\s*(password|username)\s*)?\}\}/gi

export interface ResolvedLogin {
  username: string
  password: string
}

export interface Substitution {
  /** The arguments with every reference that could be resolved replaced. */
  args: Record<string, unknown>
  /** Names referenced, whether or not they exist. */
  used: string[]
  /** Names referenced that are not saved. */
  missing: string[]
  /** Passwords that were put into the arguments, to be struck from whatever the tool prints. */
  secrets: string[]
}

/** Whether any argument, at any depth, refers to a saved login. */
export function referencesCredential(args: unknown): boolean {
  REFERENCE.lastIndex = 0
  return REFERENCE.test(JSON.stringify(args ?? {}))
}

/**
 * Replace every reference in a call's arguments.
 *
 * Walks the whole structure, because a reference can sit inside a nested object or a list as
 * easily as in a top-level string. A name that is not saved is left as written and reported, so
 * the caller can refuse the call rather than run it with a literal placeholder for a password.
 */
export function substituteCredentials(
  args: Record<string, unknown>,
  lookup: (name: string) => ResolvedLogin | null
): Substitution {
  const used = new Set<string>()
  const missing = new Set<string>()
  const secrets = new Set<string>()

  const swap = (text: string): string =>
    text.replace(REFERENCE, (whole, rawName: string, field?: string) => {
      const name = rawName.trim()
      used.add(name)
      const login = lookup(name)
      if (!login) {
        missing.add(name)
        return whole
      }
      if ((field ?? 'password').toLowerCase() === 'username') return login.username
      if (login.password) secrets.add(login.password)
      return login.password
    })

  const walk = (value: unknown): unknown => {
    if (typeof value === 'string') return swap(value)
    if (Array.isArray(value)) return value.map(walk)
    if (value && typeof value === 'object') {
      return Object.fromEntries(Object.entries(value as Record<string, unknown>).map(([k, v]) => [k, walk(v)]))
    }
    return value
  }

  return {
    args: walk(args) as Record<string, unknown>,
    used: [...used],
    missing: [...missing],
    secrets: [...secrets]
  }
}

/**
 * Strike passwords from text a tool produced.
 *
 * A command that echoes its own arguments, or a page that shows what was typed, would otherwise
 * carry the password straight back into the conversation the reference kept it out of.
 */
export function redactSecrets(text: string, secrets: string[]): string {
  let out = text
  // Longest first, so one password that contains another is not left half-struck.
  for (const secret of [...secrets].sort((a, b) => b.length - a.length)) {
    if (secret.length < 3) continue
    out = out.split(secret).join('[saved password]')
  }
  return out
}

/** Tools whose arguments carry a password in the clear, and which argument it is. */
const PASSWORD_ARGS: Record<string, string> = { save_credential: 'password' }

/**
 * A call as it may be shown and stored: with any password argument hidden.
 *
 * The call that saves a login necessarily carries the password. What is shown in the approval
 * prompt, drawn in the transcript and written to the database is this copy instead.
 */
export function maskCall(call: ToolCall): ToolCall {
  const key = PASSWORD_ARGS[call.name]
  if (!key || typeof call.args?.[key] !== 'string') return call
  return { ...call, args: { ...call.args, [key]: '••••••••' } }
}
