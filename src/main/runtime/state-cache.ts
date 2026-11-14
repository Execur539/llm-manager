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
  /**
   * Whether a loaded state can be cut back to an earlier point. A model with recurrent layers
   * cannot, so for one a conversation changed inside the saved part makes the save useless.
   */
  canTrim?: boolean
  /** The messages the state covers, in order, each with a hash of what of it the model reads. */
  messages?: StateMark[]
  /** The message a summary stood in for, up to, when it was saved: a new summary changes everything after it. */
  summaryUpto?: string | null
  /**
   * Saved part way through a reply: how much of it the state holds (characters of its answer and
   * of its thinking), and a hash of exactly that, to tell later whether the finished reply still
   * begins that way. The reply is the message after the last of `messages`.
   */
  partial?: PartialMark | null
}

export interface PartialMark {
  content: number
  reasoning: number
  hash: string
}

/** What a save covers, as given to it. */
export interface StateCovers {
  messages: StateMark[]
  summaryUpto: string | null
  canTrim: boolean
  partial?: PartialMark | null
}

/** A message a save covers: its id, and a hash of what the model reads of it. */
export interface StateMark {
  id: string
  hash: string
}

/**
 * How many of the conversation's messages, from the start, the saved state still holds as they are.
 *
 * Everything said after a save is simply not covered: it is read again on top of the loaded state,
 * never lost. A message changed, removed or rewound inside the saved part ends the covered run
 * there. A model that can cut its state back reuses what comes before; one with recurrent layers
 * cannot, so for it the save is no use at all (`dropped`).
 */
export function coverage(meta: StateMeta, current: StateMark[], summaryUpto: string | null): { count: number; dropped: boolean } {
  const saved = meta.messages
  if (!saved?.length) return { count: 0, dropped: false }
  let k = 0
  if ((meta.summaryUpto ?? null) === summaryUpto) {
    while (k < saved.length && k < current.length && saved[k].id === current[k].id && saved[k].hash === current[k].hash) k++
  }
  if (k === saved.length) return { count: k, dropped: false }
  return meta.canTrim ? { count: k, dropped: false } : { count: 0, dropped: true }
}

/** How a save is going, while it runs: the server writing the file, then (protected) encrypting it. */
export type SaveProgress = { stage: 'writing' | 'encrypting'; done: number; total: number | null }

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

/** Whether anything in the conversation is not text. */
export function hasMedia(messages: ChatMessage[]): boolean {
  return messages.some((m) => typeof m.content !== 'string' && m.content.some((p) => p.type !== 'text'))
}

/**
 * How the server marks a picture, sound or video in a rendered prompt.
 *
 * Pinned when the server starts (LLAMA_MEDIA_MARKER). Left to itself llama.cpp picks a new random
 * marker every launch, so a conversation with pictures in it rendered differently after every
 * restart and could never be matched to its save.
 */
export const MEDIA_MARKER = '<__llmm_media__>'

interface MediaItem {
  /** The bytes, base64, as the server is sent them. */
  data: string
  /** Their SHA-256, which is also how the server itself tells one picture from another. */
  hash: string
}

/**
 * The pictures, sounds and videos in a conversation, in the order its prompt meets them. Null when
 * one is not bytes the app has (a picture given as a web address).
 */
function mediaOf(messages: ChatMessage[]): MediaItem[] | null {
  const out: MediaItem[] = []
  for (const m of messages) {
    if (typeof m.content === 'string') continue
    for (const p of m.content) {
      let data: string | undefined
      if (p.type === 'image_url') {
        const url = p.image_url?.url ?? ''
        const at = url.indexOf('base64,')
        if (!url.startsWith('data:') || at < 0) return null
        data = url.slice(at + 'base64,'.length)
      } else if (p.type === 'input_audio') {
        data = p.input_audio?.data
      } else if (p.type === 'input_video') {
        data = p.input_video?.data
      } else {
        continue
      }
      if (!data) return null
      out.push({ data, hash: crypto.createHash('sha256').update(Buffer.from(data, 'base64')).digest('hex') })
    }
  }
  return out
}

/** How many pictures, sounds and videos a piece of rendered prompt holds. */
const markersIn = (text: string): number => text.split(MEDIA_MARKER).length - 1

/**
 * What identifies the start of a conversation: its rendered text and, since every picture renders
 * as the same marker, the pictures in it. Just the text, as it always was, when there are none.
 */
function stamp(text: string, media: MediaItem[]): string {
  const n = markersIn(text)
  return n ? sha(`${text}\u0000${media.slice(0, n).map((m) => m.hash).join(',')}`) : sha(text)
}

/**
 * What brings the server to exactly `text`. With pictures in it, they go along with it: sent as
 * plain text the markers would be read as words, and the server would throw away the pictures it
 * holds to read them.
 */
function promptFor(text: string, media: MediaItem[]): unknown {
  const n = markersIn(text)
  return n ? { prompt_string: text, multimodal_data: media.slice(0, n).map((m) => m.data) } : text
}

/** Why a conversation's media cannot be saved with it, or null when they can. */
function mediaProblem(media: MediaItem[] | null, prefix: string): string | null {
  if (!media) return 'a picture in it is a web address rather than a file the app has'
  if (markersIn(prefix) !== media.length) return 'the server did not mark its pictures the way the app expects; load the model again'
  return null
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
  if (!meta || meta.fingerprint !== o.fingerprint) return null
  const media = mediaOf(o.req.messages)
  if (!media) return null
  const prefix = await renderPrefix(o.base, o.req, o.signal)
  return prefix.length === meta.prefixChars && stamp(prefix, media) === meta.prefixHash ? meta : null
}

/**
 * The saved state, of any conversation, that holds the longest start of this one: the instructions
 * every conversation begins with, or the history a fork shares with the conversation it came from.
 * Only states saved under the same load, and only those `allow` lets through. Null when none does.
 */
export function longestReusable(
  dir: string,
  fingerprint: string,
  prefix: string,
  messages: ChatMessage[],
  allow: (meta: StateMeta) => boolean
): StateMeta | null {
  const media = mediaOf(messages) ?? []
  let best: StateMeta | null = null
  for (const s of listStates(dir)) {
    if (s.fingerprint !== fingerprint || s.prefixChars > prefix.length || (best && s.prefixChars <= best.prefixChars)) continue
    if (!allow(s) || stamp(prefix.slice(0, s.prefixChars), media) !== s.prefixHash) continue
    best = s
  }
  return best
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
  /** What it covers, kept in its record so the window can show which messages are saved. */
  covers?: StateCovers
  /** About how large the file will be, from earlier saves, for reporting how far writing it has got. */
  expectedBytes?: number | null
  onProgress?: (p: SaveProgress) => void
}): Promise<StateResult> {
  if (!o.req.messages.length) return { ok: false, why: 'the conversation is empty' }
  try {
    const media = mediaOf(o.req.messages)
    const prefix = await renderPrefix(o.base, o.req, o.signal)
    if (!prefix) return { ok: false, why: 'the conversation rendered to nothing' }
    const problem = mediaProblem(media, prefix)
    if (problem || !media) return { ok: false, why: problem ?? 'its media could not be read' }
    // Leaves the slot holding exactly this text: nothing after it for a later prompt to undo.
    await post(o.base, '/completion', { prompt: promptFor(prefix, media), n_predict: 0, cache_prompt: true, stream: false }, o.signal)
    return await writeState(o, prefix, media)
  } catch (err) {
    clearPartials(o.dir, o.key)
    return { ok: false, why: err instanceof Error ? err.message : String(err) }
  }
}

/**
 * Save a conversation part way through a reply: `sent` is the request that resumes it, the
 * conversation with the reply so far as its final message.
 *
 * The server is brought to exactly that text first, as an ordinary save brings it to the end of
 * the conversation. Stopping a stream leaves it holding a token or two the window never received,
 * and a state holding those would not match the resumed reply, nor the finished one later. The
 * record vouches for that text, so a later load checks the conversation still begins with it.
 */
export async function saveAsIs(o: {
  base: string
  dir: string
  key: string
  fingerprint: string
  sent: StateRequest & { continueFinal?: boolean }
  signal?: AbortSignal
  seal?: Buffer
  covers?: StateCovers
  expectedBytes?: number | null
  onProgress?: (p: SaveProgress) => void
}): Promise<StateResult> {
  if (!o.sent.messages.length) return { ok: false, why: 'the conversation is empty' }
  try {
    const media = mediaOf(o.sent.messages)
    const prefix = await renderSent(o.base, o.sent, o.signal)
    if (!prefix) return { ok: false, why: 'the conversation rendered to nothing' }
    const problem = mediaProblem(media, prefix)
    if (problem || !media) return { ok: false, why: problem ?? 'its media could not be read' }
    await post(o.base, '/completion', { prompt: promptFor(prefix, media), n_predict: 0, cache_prompt: true, stream: false }, o.signal)
    return await writeState(o, prefix, media)
  } catch (err) {
    clearPartials(o.dir, o.key)
    return { ok: false, why: err instanceof Error ? err.message : String(err) }
  }
}

/** The prompt a request sent, as the server rendered it. */
async function renderSent(base: string, req: StateRequest & { continueFinal?: boolean }, signal?: AbortSignal): Promise<string> {
  const body: Record<string, unknown> = { messages: req.messages }
  if (req.tools?.length) {
    body.tools = req.tools.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: t.parameters }
    }))
  }
  if (req.reasoningEffort) body.reasoning_effort = req.reasoningEffort
  if (req.chatTemplateKwargs && Object.keys(req.chatTemplateKwargs).length) body.chat_template_kwargs = req.chatTemplateKwargs
  if (req.continueFinal) {
    body.add_generation_prompt = false
    body.continue_final_message = true
  }
  const out = await post(base, '/apply-template', body, signal)
  if (typeof out.prompt !== 'string') throw new Error('the server did not return a rendered prompt')
  return out.prompt
}

function clearPartials(dir: string, key: string): void {
  fs.rmSync(path.join(dir, `${fileStem(key)}.next.bin`), { force: true })
  fs.rmSync(path.join(dir, `${fileStem(key)}.next.lk`), { force: true })
}

/** Write what the slot holds to the conversation's file and its record, `prefix` being the text it holds. */
async function writeState(
  o: {
    base: string
    dir: string
    key: string
    fingerprint: string
    signal?: AbortSignal
    seal?: Buffer
    covers?: StateCovers
    expectedBytes?: number | null
    onProgress?: (p: SaveProgress) => void
  },
  prefix: string,
  media: MediaItem[]
): Promise<StateResult> {
  {
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
    // The server says nothing until the file is written, so how far it has got is read off the file as it grows.
    const watch = o.onProgress
      ? setInterval(() => {
          try {
            o.onProgress?.({ stage: 'writing', done: fs.statSync(next).size, total: o.expectedBytes ?? null })
          } catch {
            /* not started yet */
          }
        }, 250)
      : null
    let saved: Record<string, unknown>
    try {
      saved = await post(o.base, '/slots/0?action=save', { filename: `${stem}.next.bin` }, o.signal)
    } finally {
      if (watch) clearInterval(watch)
    }
    let written = next
    if (o.seal) {
      /*
       * A protected conversation's state holds its words, as tokens. The server can only write it
       * plain, so it is encrypted straight away and the plain file removed: it exists for as long
       * as the save and the encryption take, never once the save is over.
       */
      written = path.join(o.dir, `${stem}.next.lk`)
      try {
        await sealFile(o.seal, next, written, (done, total) => o.onProgress?.({ stage: 'encrypting', done, total }))
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
      prefixHash: stamp(prefix, media),
      tokens: Number(saved.n_saved ?? 0),
      bytes: Number(saved.n_written ?? 0),
      fileBytes,
      savedAt: Date.now(),
      usedAt: Date.now(),
      ...(o.covers
        ? { canTrim: o.covers.canTrim, messages: o.covers.messages, summaryUpto: o.covers.summaryUpto, partial: o.covers.partial ?? null }
        : {})
    }
    // The record is written last and whole, so a state without one is never taken for a good one.
    const tmp = `${metaPath(o.dir, o.key)}.tmp`
    fs.writeFileSync(tmp, JSON.stringify(meta))
    fs.renameSync(tmp, metaPath(o.dir, o.key))
    return { ok: true, meta }
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
  /**
   * Another conversation's state, borrowed for the start this one shares with it. Never removed
   * or changed here, whatever is found: it is not this conversation's to discard.
   */
  borrowed?: boolean
}): Promise<StateResult> {
  const meta = readMeta(o.dir, o.key)
  if (!meta) return { ok: false, why: 'nothing is saved for this conversation' }
  // Kept, not removed: the user saved it, and it loads again once that model is back.
  if (meta.fingerprint !== o.fingerprint) return { ok: false, why: 'it was saved under a different model or different load settings' }
  try {
    const media = mediaOf(o.req.messages) ?? []
    const prefix = await renderPrefix(o.base, o.req, o.signal)
    const begins = prefix.length >= meta.prefixChars && stamp(prefix.slice(0, meta.prefixChars), media) === meta.prefixHash
    /*
     * Changed inside the saved part. A model that can cut its state back still gains: the server
     * keeps what matches the new prompt and reads the rest. One with recurrent layers cannot, and
     * the save is of no use to it any more.
     */
    if (!begins && (o.borrowed || !meta.canTrim)) {
      if (!o.borrowed) removeState(o.dir, o.key)
      return { ok: false, why: 'the conversation no longer begins the way it did when it was saved' }
    }
    const stem = fileStem(o.key)
    const sealed = await isSealedFile(path.join(o.dir, `${stem}.bin`))
    if (sealed !== !!o.seal) {
      // Encrypted with no key to open it, or a protected conversation's state left plain: neither is used.
      if (o.borrowed) return { ok: false, why: 'it is encrypted' }
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
      if (!o.borrowed) removeState(o.dir, o.key)
      return { ok: false, why: `the server restored ${String(loaded.n_restored)} tokens where ${meta.tokens} were saved` }
    }
    const used: StateMeta = { ...meta, usedAt: Date.now() }
    fs.writeFileSync(metaPath(o.dir, o.key), JSON.stringify(used))
    return { ok: true, meta: used }
  } catch (err) {
    return { ok: false, why: err instanceof Error ? err.message : String(err) }
  }
}
