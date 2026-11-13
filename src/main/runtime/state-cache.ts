/**
 * What the model has read of a conversation, kept on disk.
 *
 * The server holds one conversation at a time, and whatever it holds is gone when another one
 * takes its place, the model is reloaded or the app restarts; coming back then means reading the
 * whole conversation again. llama-server can write a slot's state to a file and load it back,
 * which turns that reading into a file read.
 *
 * Two things make it less simple than save and load.
 *
 * A model with recurrent layers cannot step backwards from a loaded state: the checkpoints it
 * would step back to are not in the file. So the state that is saved has to end exactly where
 * every later prompt for the conversation continues from, with nothing after it to undo. That
 * point is the start of the next user message. It is found by rendering the conversation twice
 * with two different stand-in messages and keeping what the two have in common, and the server
 * is brought to exactly that text before the slot is saved.
 *
 * And a state is only worth loading into the server it came from. Each file carries a
 * fingerprint of the model and how it was loaded, and a hash of the text it holds; a state whose
 * fingerprint differs, or whose text is no longer how the conversation begins, is left alone.
 *
 * Nothing here imports Electron or the runtime, so it can be exercised against a real server
 * from a script.
 */

import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
import type { ToolDefinition } from '@shared/types'
import type { ChatMessage } from './llama'
import { isSealedFile, openFile, sealFile } from '../protect/cipher'

/** What decides how a conversation renders: its messages and everything else the template sees. */
export interface StateRequest {
  messages: ChatMessage[]
  tools?: ToolDefinition[]
  reasoningEffort?: string
  chatTemplateKwargs?: Record<string, unknown>
}

export interface StateMeta {
  key: string
  fingerprint: string
  /** Length of the rendered text the state holds, and its hash. */
  prefixChars: number
  prefixHash: string
  tokens: number
  bytes: number
  /**
   * The state file's size on disk when it was saved. A file of any other size is not the one this
   * record describes (a save cut off part way, say) and is never loaded. Absent in older records.
   */
  fileBytes?: number
  savedAt: number
  usedAt: number
}

/**
 * When a conversation is worth saving again at a quiet moment: what checkpointDue weighs.
 *
 * Saving rewrites the whole state, which for a long conversation is gigabytes, so it is not done
 * after every reply. A short conversation is never saved this way (reading it again takes about
 * as long as loading it would). A longer one is saved once enough is new to be worth a file, in
 * proportion to its length, so a 500,000-token conversation is not rewritten for every line; and
 * whatever has changed is saved at least every ten minutes. Switching away, unloading, quitting
 * and the machine going to sleep save it regardless.
 */
export const CHECKPOINT = {
  /** Below this many tokens a conversation is not worth a file. */
  minTokens: 1024,
  /** Less new than this is never worth rewriting the file for. */
  minNew: 256,
  /** Enough new to save straight away: this share of the conversation, but at least `floor`. */
  share: 0.03,
  floor: 1024,
  /** Anything new at all is saved once the file is this old. */
  maxAgeMs: 10 * 60_000
}

/**
 * Whether to save a conversation now, given how long it is (tokens, as the server last counted
 * them, or null when it never said) and what is already saved of it.
 */
export function checkpointDue(total: number | null, saved: { tokens: number; savedAt: number } | null, now: number): boolean {
  if (total === null) return !saved || now - saved.savedAt >= CHECKPOINT.maxAgeMs
  if (total < CHECKPOINT.minTokens) return false
  if (!saved) return true
  const fresh = total - saved.tokens
  if (fresh < CHECKPOINT.minNew) return false
  if (fresh >= Math.max(CHECKPOINT.floor, total * CHECKPOINT.share)) return true
  return now - saved.savedAt >= CHECKPOINT.maxAgeMs
}

/** Whether a state file on disk is the whole of what its record describes. */
function fileMatches(dir: string, meta: StateMeta): boolean {
  try {
    const size = fs.statSync(path.join(dir, `${fileStem(meta.key)}.bin`)).size
    return meta.fileBytes === undefined || size === meta.fileBytes
  } catch {
    return false
  }
}

export type StateResult = { ok: true; meta: StateMeta } | { ok: false; why: string }

const sha = (text: string): string => crypto.createHash('sha256').update(text).digest('hex')

/** A file name for a conversation; the server takes a bare name and nothing that looks like a path. */
export function fileStem(key: string): string {
  return `state-${sha(key).slice(0, 20)}`
}

/** Whether anything in the conversation is not text. A raw prompt cannot carry media. */
export function hasMedia(messages: ChatMessage[]): boolean {
  return messages.some((m) => typeof m.content !== 'string' && m.content.some((p) => p.type !== 'text'))
}

async function post(base: string, url: string, body: unknown, signal?: AbortSignal): Promise<Record<string, unknown>> {
  const res = await fetch(base + url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal
  })
  const text = await res.text()
  let json: Record<string, unknown>
  try {
    json = JSON.parse(text) as Record<string, unknown>
  } catch {
    throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`)
  }
  if (!res.ok || json.error) throw new Error(`HTTP ${res.status}: ${JSON.stringify(json.error ?? json).slice(0, 300)}`)
  return json
}

async function render(base: string, req: StateRequest, standIn: string, signal?: AbortSignal): Promise<string> {
  const body: Record<string, unknown> = { messages: [...req.messages, { role: 'user', content: standIn }] }
  if (req.tools?.length) {
    body.tools = req.tools.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: t.parameters }
    }))
  }
  if (req.reasoningEffort) body.reasoning_effort = req.reasoningEffort
  if (req.chatTemplateKwargs && Object.keys(req.chatTemplateKwargs).length) body.chat_template_kwargs = req.chatTemplateKwargs
  const out = await post(base, '/apply-template', body, signal)
  if (typeof out.prompt !== 'string') throw new Error('the server did not return a rendered prompt')
  return out.prompt
}

/**
 * The rendered conversation up to where its next user message begins.
 *
 * Whatever two renderings with different next messages share is, by construction, the part no
 * next message can change.
 */
export async function renderPrefix(base: string, req: StateRequest, signal?: AbortSignal): Promise<string> {
  const [a, b] = await Promise.all([render(base, req, 'AAAAAAAA', signal), render(base, req, 'BBBBBBBB', signal)])
  let n = 0
  while (n < a.length && n < b.length && a[n] === b[n]) n++
  return a.slice(0, n)
}

/**
 * The saved state, when it holds this conversation exactly as it is now: saved under the same load,
 * with nothing said since. Null when there is none, or the conversation has moved on from it.
 */
export async function currentState(o: {
  base: string
  dir: string
  key: string
  fingerprint: string
  req: StateRequest
  signal?: AbortSignal
}): Promise<StateMeta | null> {
  const meta = readMeta(o.dir, o.key)
  if (!meta || meta.fingerprint !== o.fingerprint || hasMedia(o.req.messages)) return null
  const prefix = await renderPrefix(o.base, o.req, o.signal)
  return prefix.length === meta.prefixChars && sha(prefix) === meta.prefixHash ? meta : null
}

function metaPath(dir: string, key: string): string {
  return path.join(dir, `${fileStem(key)}.json`)
}

export function readMeta(dir: string, key: string): StateMeta | null {
  try {
    const meta = JSON.parse(fs.readFileSync(metaPath(dir, key), 'utf8')) as StateMeta
    return meta.key === key && fileMatches(dir, meta) ? meta : null
  } catch {
    return null
  }
}

export function removeState(dir: string, key: string): void {
  for (const ext of ['.bin', '.json', '.next.bin', '.next.lk', '.open.bin']) fs.rmSync(path.join(dir, fileStem(key) + ext), { force: true })
}

export function listStates(dir: string): StateMeta[] {
  let names: string[]
  try {
    names = fs.readdirSync(dir)
  } catch {
    return []
  }
  const out: StateMeta[] = []
  for (const name of names) {
    if (!/^state-[0-9a-f]{20}\.json$/.test(name)) continue
    try {
      const meta = JSON.parse(fs.readFileSync(path.join(dir, name), 'utf8')) as StateMeta
      if (fileMatches(dir, meta)) out.push(meta)
    } catch {
      /* a half-written record is no record */
    }
  }
  return out
}

/**
 * Bring the folder under its budget, least recently used first, never touching a pinned one.
 * Returns the conversations whose state was removed.
 */
export function prune(dir: string, budgetBytes: number, pinned: ReadonlySet<string>): string[] {
  const states = listStates(dir).sort((a, b) => a.usedAt - b.usedAt)
  let total = states.reduce((n, s) => n + s.bytes, 0)
  const removed: string[] = []
  for (const s of states) {
    if (total <= budgetBytes) break
    if (pinned.has(s.key)) continue
    removeState(dir, s.key)
    total -= s.bytes
    removed.push(s.key)
  }
  return removed
}

/**
 * Save what the server holds of a conversation.
 *
 * Only worth calling while the server is holding that conversation: the first step brings it to
 * the exact end of the rendered text, which is a few tokens of work then and a full reading
 * otherwise.
 */
export async function saveState(o: {
  base: string
  dir: string
  key: string
  fingerprint: string
  req: StateRequest
  signal?: AbortSignal
  /** A protected conversation's key: the state is stored encrypted under it. */
  seal?: Buffer
}): Promise<StateResult> {
  if (!o.req.messages.length) return { ok: false, why: 'the conversation is empty' }
  if (hasMedia(o.req.messages)) return { ok: false, why: 'the conversation contains images, audio or video' }
  try {
    const prefix = await renderPrefix(o.base, o.req, o.signal)
    if (!prefix) return { ok: false, why: 'the conversation rendered to nothing' }
    // Leaves the slot holding exactly this text: nothing after it for a later prompt to undo.
    await post(o.base, '/completion', { prompt: prefix, n_predict: 0, cache_prompt: true, stream: false }, o.signal)
    const stem = fileStem(o.key)
    /*
     * Written beside the last good state, and only put in its place once it is whole.
     *
     * The server writes the file itself, as it goes. Saving straight over the old one meant a crash
     * or a power cut part way through left neither: the old state half overwritten, under a record
     * that still vouched for it. Now an interrupted save leaves the previous state as it was.
     */
    const next = path.join(o.dir, `${stem}.next.bin`)
    fs.rmSync(next, { force: true })
    const saved = await post(o.base, '/slots/0?action=save', { filename: `${stem}.next.bin` }, o.signal)
    let written = next
    if (o.seal) {
      /*
       * A protected conversation's state holds its words, as tokens. The server can only write it
       * plain, so it is encrypted straight away and the plain file removed: it exists for as long
       * as the save and the encryption take, never once the save is over.
       */
      written = path.join(o.dir, `${stem}.next.lk`)
      try {
        await sealFile(o.seal, next, written)
      } finally {
        fs.rmSync(next, { force: true })
      }
    }
    const fileBytes = fs.statSync(written).size
    fs.renameSync(written, path.join(o.dir, `${stem}.bin`))
    const meta: StateMeta = {
      key: o.key,
      fingerprint: o.fingerprint,
      prefixChars: prefix.length,
      prefixHash: sha(prefix),
      tokens: Number(saved.n_saved ?? 0),
      bytes: Number(saved.n_written ?? 0),
      fileBytes,
      savedAt: Date.now(),
      usedAt: Date.now()
    }
    // The record is written last and whole, so a state without one is never taken for a good one.
    const tmp = `${metaPath(o.dir, o.key)}.tmp`
    fs.writeFileSync(tmp, JSON.stringify(meta))
    fs.renameSync(tmp, metaPath(o.dir, o.key))
    return { ok: true, meta }
  } catch (err) {
    fs.rmSync(path.join(o.dir, `${fileStem(o.key)}.next.bin`), { force: true })
    fs.rmSync(path.join(o.dir, `${fileStem(o.key)}.next.lk`), { force: true })
    return { ok: false, why: err instanceof Error ? err.message : String(err) }
  }
}

/**
 * Load a conversation's saved state into the server, if it still is how the conversation begins.
 *
 * A conversation that has grown since it was saved still begins with the saved text, and the
 * state is still good: the next prompt adds to it. One that was edited or rewound does not, and
 * its state is removed.
 */
export async function restoreState(o: {
  base: string
  dir: string
  key: string
  fingerprint: string
  req: StateRequest
  signal?: AbortSignal
  /** A protected conversation's key, when its state is stored encrypted. */
  seal?: Buffer
}): Promise<StateResult> {
  const meta = readMeta(o.dir, o.key)
  if (!meta) return { ok: false, why: 'nothing is saved for this conversation' }
  if (meta.fingerprint !== o.fingerprint) {
    removeState(o.dir, o.key)
    return { ok: false, why: 'it was saved under a different model or different load settings' }
  }
  if (hasMedia(o.req.messages)) return { ok: false, why: 'the conversation contains images, audio or video' }
  try {
    const prefix = await renderPrefix(o.base, o.req, o.signal)
    if (prefix.length < meta.prefixChars || sha(prefix.slice(0, meta.prefixChars)) !== meta.prefixHash) {
      removeState(o.dir, o.key)
      return { ok: false, why: 'the conversation no longer begins the way it did when it was saved' }
    }
    const stem = fileStem(o.key)
    const sealed = await isSealedFile(path.join(o.dir, `${stem}.bin`))
    if (sealed !== !!o.seal) {
      // Encrypted with no key to open it, or a protected conversation's state left plain: neither is used.
      removeState(o.dir, o.key)
      return { ok: false, why: sealed ? 'it is encrypted and the conversation is not protected' : 'it was saved before the conversation was protected' }
    }
    /*
     * The server reads only plain files, so an encrypted one is opened into a file beside it for
     * the length of the load and removed the moment it is read.
     */
    const file = sealed ? `${stem}.open.bin` : `${stem}.bin`
    if (sealed && o.seal) await openFile(o.seal, path.join(o.dir, `${stem}.bin`), path.join(o.dir, file))
    let loaded: Record<string, unknown>
    try {
      loaded = await post(o.base, '/slots/0?action=restore', { filename: file }, o.signal)
    } finally {
      if (sealed) fs.rmSync(path.join(o.dir, file), { force: true })
    }
    if (Number(loaded.n_restored ?? -1) !== meta.tokens) {
      // The file and its record disagree, so neither can be trusted again.
      removeState(o.dir, o.key)
      return { ok: false, why: `the server restored ${String(loaded.n_restored)} tokens where ${meta.tokens} were saved` }
    }
    const used: StateMeta = { ...meta, usedAt: Date.now() }
    fs.writeFileSync(metaPath(o.dir, o.key), JSON.stringify(used))
    return { ok: true, meta: used }
  } catch (err) {
    return { ok: false, why: err instanceof Error ? err.message : String(err) }
  }
}
