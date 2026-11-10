/**
 * Tools for the agent's saved logins.
 *
 * The password is deliberately not something these hand to the model in the ordinary course.
 * `list_credentials` says what is saved and how to refer to it; the reference is filled in by the
 * loop when a tool call that uses it runs (see credential-refs). `reveal_credential` exists for
 * the case where the user asks to be told a password, and is execute-tier so it always asks.
 *
 * None of these are offered to a remote session: someone signed in over the web interface is not
 * necessarily the person whose passwords these are.
 */

import type { Tool } from './base'
import { schema, str } from './base'
import { deleteLogin, listLogins, resolveLogin, saveLogin } from '../credentials'

/** Tools a remote session is never given, and never allowed to run. */
export const LOCAL_ONLY_TOOLS = new Set(['save_credential', 'list_credentials', 'reveal_credential', 'delete_credential'])

const HOW_TO_USE =
  'To use a password, write {{credential:NAME}} where it belongs in the arguments of a tool call that ' +
  'types, fills in or runs something; it is replaced with the real password when the call runs. ' +
  '{{credential:NAME:username}} gives the username the same way.'

const saveCredential: Tool = {
  name: 'save_credential',
  description:
    'Save a username and password under a name, to use again in later conversations. Saving under ' +
    'a name that already exists replaces it. Use when the user gives you a login to remember.',
  tier: 'write',
  parameters: schema(
    {
      name: str('Short name for this login, such as "github" or "router"'),
      username: str('Username, email address or account name'),
      password: str('The password'),
      url: str('Where it is used: a site address or an app name'),
      notes: str('Anything else worth knowing, such as which account this is')
    },
    ['name', 'password']
  ),
  async run(args) {
    const saved = saveLogin({
      name: String(args.name ?? ''),
      username: args.username === undefined ? '' : String(args.username),
      password: String(args.password ?? ''),
      url: args.url === undefined ? null : String(args.url),
      notes: args.notes === undefined ? null : String(args.notes)
    })
    return `Saved the login "${saved.name}"${saved.username ? ` (username ${saved.username})` : ''}. ${HOW_TO_USE.replace(/NAME/g, saved.name)}`
  }
}

const listCredentials: Tool = {
  name: 'list_credentials',
  description:
    'List the saved logins: their names, usernames, where they are used and any notes. Passwords ' +
    'are not shown. ' + HOW_TO_USE,
  tier: 'read',
  parameters: schema({ filter: str('Only logins whose name, username, url or notes contain this') }),
  async run(args) {
    const needle = args.filter ? String(args.filter).toLowerCase() : null
    const logins = listLogins().filter(
      (l) => !needle || `${l.name} ${l.username} ${l.url ?? ''} ${l.notes ?? ''}`.toLowerCase().includes(needle)
    )
    if (!logins.length) return needle ? `No saved login matches "${args.filter}".` : 'No logins are saved yet.'
    const lines = logins.map(
      (l) =>
        `- ${l.name}: username ${l.username || '(none)'}` +
        (l.url ? `, used at ${l.url}` : '') +
        (l.notes ? `, notes: ${l.notes}` : '')
    )
    return `${lines.join('\n')}\n\n${HOW_TO_USE}`
  }
}

const revealCredential: Tool = {
  name: 'reveal_credential',
  description:
    'Show the password of a saved login in plain text. Only for when the user asks to be told a ' +
    'password. To sign in somewhere, use {{credential:NAME}} in the tool call instead, which ' +
    'needs no reveal.',
  tier: 'execute',
  parameters: schema({ name: str('Name of the saved login') }, ['name']),
  async run(args) {
    const login = resolveLogin(String(args.name ?? ''))
    if (!login) return `No login is saved under "${args.name}". list_credentials shows what is saved.`
    return `Login "${args.name}": username ${login.username || '(none)'}, password ${login.password}`
  }
}

const deleteCredential: Tool = {
  name: 'delete_credential',
  description: 'Delete a saved login.',
  tier: 'write',
  parameters: schema({ name: str('Name of the saved login') }, ['name']),
  async run(args) {
    return deleteLogin(String(args.name ?? ''))
      ? `Deleted the login "${args.name}".`
      : `No login is saved under "${args.name}".`
  }
}

export const credentialTools: Tool[] = [saveCredential, listCredentials, revealCredential, deleteCredential]
