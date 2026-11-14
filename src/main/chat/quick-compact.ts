/**
 * Quick compact: shrinking a conversation with code alone, no model.
 *
 * The other compaction asks the model for a summary, which takes its writing time and can fail.
 * This one rewrites how the older part of a conversation is *sent*, by rules, in an instant. The
 * transcript on disk is never touched: what is stored is only which stretches are shortened and
 * how hard, and everything left out stays reachable through the recall tools (see recall.ts),
 * arguments and pictures included, which is what makes shortening this much safe.
 *
 * How hard is the user's choice, from five strengths whose resulting sizes are shown first:
 *
 *   1  thinking dropped, tool output tidied (colour codes, repeated lines, endless lines)
 *   2  + reads and commands that a later identical one superseded are dropped
 *   3  + every tool result becomes one line saying what it was; attachments and images go
 *   4  + each turn becomes the user's message, an account of what followed, and the final answer
 *   5  + answers become outlines, long messages are cut to their start and end
 *
 * What a strength never touches: what the user said, including their answers to the agent's
 * questions; which files were changed; what failed; and the short remarks the model made on the
 * way. At strengths 4 and 5 the newer half of what is compacted is done one strength gentler.
 *
 * A conversation is compacted in stretches (segments), each with its own strength and each
 * rendered the same way whatever comes after it. Compacting again adds a stretch and leaves the
 * earlier ones exactly as they were, however hard or gently they were done. A stretch may end
 * part way through a long turn, so one request that ran two hundred tool calls can be compacted.
 *
 * Pure: no Electron, no database, so it is checked from a script.
 */

import type { AgentMessage, ToolCall } from '@shared/types'

export interface QuickSegment {
  /** The last message of this stretch. */
  upto: string
  /** How hard it was shortened: 1 to 5. */
  level: number
}

export interface QuickSpec {
  v: 2
  segments: QuickSegment[]
}

export const QUICK_LEVELS = 5
/** The strength suggested is the gentlest that brings the whole conversation under this share of the window. */
export const QUICK_TARGET_SHARE = 0.1
/** The most recent messages stay word for word: this share of the window, and never more than the cap. */
export const QUICK_KEEP_SHARE = 0.04
export const QUICK_KEEP_MAX_TOKENS = 32_000

const tokens = (text: string | undefined | null): number => Math.ceil((text?.length ?? 0) / 4)
const IMAGE_TOKENS = 1200

/** What a message costs to send, near enough to choose a strength by. */
export function costOf(m: AgentMessage, sentText?: (id: string) => string | undefined): number {
  const body = (m.role === 'user' ? sentText?.(m.id) : undefined) ?? m.content
  const calls = m.role === 'tool' ? tokens(JSON.stringify(m.toolCalls?.[0]?.args ?? {})) + 12 : 0
  const pictures = (m.attachments?.filter((a) => a.kind === 'image').length ?? 0) * IMAGE_TOKENS + (m.toolImages ?? 0) * IMAGE_TOKENS
  return tokens(body) + tokens(m.reasoning) + calls + pictures
}

// ------------------------------------------------------------------ text

/** Tool output tidied without losing anything a reader would miss. */
export function squeeze(text: string): string {
  // eslint-disable-next-line no-control-regex
  const plain = text.replace(/\u001b\[[0-9;?]*[ -/]*[@-~]/g, '').replace(/\r/g, '')
  const out: string[] = []
  let last: string | null = null
  let repeats = 0
  let blanks = 0
  const flush = (): void => {
    if (repeats > 0) out.push(`  (the line above, ${repeats} more time${repeats === 1 ? '' : 's'})`)
    repeats = 0
  }
  for (const raw of plain.split('\n')) {
    const line = raw.length > 400 ? `${raw.slice(0, 300)} … (${raw.length} characters)` : raw
    if (!line.trim()) {
      flush()
      last = null
      if (++blanks <= 1) out.push('')
      continue
    }
    blanks = 0
    if (line === last) {
      repeats++
      continue
    }
    flush()
    out.push(line)
    last = line
  }
  flush()
  return out.join('\n')
}

/** The first `n` sentences of a paragraph. */
function sentences(text: string, n: number): string {
  const parts = text.match(/[^.!?]*[.!?]+(?=\s|$)|[^.!?]+$/g) ?? [text]
  return parts.slice(0, n).join('').trim()
}

/**
 * A markdown answer as its outline: headings, the start of each paragraph, list items, and code
 * blocks whole when short and by their opening lines when long.
 */
export function outline(text: string): string {
  const out: string[] = []
  let fence: { open: string; lines: string[] } | null = null
  let paragraph: string[] = []
  const flush = (): void => {
    if (!paragraph.length) return
    const two = sentences(paragraph.join(' '), 2)
    out.push(two.length > 360 ? `${two.slice(0, 340)}…` : two)
    paragraph = []
  }
  for (const line of text.split('\n')) {
    if (/^\s*```/.test(line)) {
      if (fence) {
        const body = fence.lines
        out.push(fence.open, ...(body.length <= 12 ? body : [...body.slice(0, 6), `… (${body.length - 6} more lines)`]), '```')
        fence = null
      } else {
        flush()
        fence = { open: line.trim(), lines: [] }
      }
      continue
    }
    if (fence) {
      fence.lines.push(line)
      continue
    }
    const t = line.trim()
    if (!t) {
      flush()
      continue
    }
    if (/^(#{1,6}\s|[-*+]\s|\d+[.)]\s|>|\|)/.test(t)) {
      flush()
      // Headings whole; list items, quotes and table rows by their start.
      out.push(/^#/.test(t) ? t : t.length > 170 ? `${t.slice(0, 160)}…` : t)
      continue
    }
    paragraph.push(t)
  }
  flush()
  if (fence) out.push(fence.open, ...fence.lines.slice(0, 6), '```')
  return out.join('\n')
}

function headTail(text: string, max: number): string {
  if (text.length <= max) return text
  const head = Math.floor(max * 0.65)
  const tail = max - head - 40
  return `${text.slice(0, head)}\n… (${text.length - head - tail} characters left out) …\n${text.slice(-tail)}`
}

const oneLine = (text: string, max: number): string => {
  const t = text.replace(/\s+/g, ' ').trim()
  return t.length > max ? `${t.slice(0, max - 1)}…` : t
}

// ------------------------------------------------------------------ tool calls

/** The argument that says what a call was about. */
export function callDetail(call: ToolCall | undefined): string {
  const a = (call?.args ?? {}) as Record<string, unknown>
  for (const key of ['path', 'name', 'command', 'query', 'url', 'pattern', 'question', 'id']) {
    if (typeof a[key] === 'string' && a[key]) return oneLine(a[key] as string, 90)
  }
  return ''
}

/** Tools whose results are the user speaking: never shortened, at any strength. */
const USER_SPEAKS = new Set(['ask_user'])
/** Tools that change files. */
const CHANGES = /^(write_file|edit_file|delete_file|move_file|copy_file|apply_patch|create_file)$/
const isChange = (name: string): boolean => CHANGES.test(name) || /(^|_)(write|edit|delete|patch)(_|$)/.test(name)

interface Wording {
  /** Whether the model has the recall tools, and so should be told how to get things back. */
  recall: boolean
}

const back = (w: Wording, id: string): string => (w.recall ? ` recall_read("${id}") shows it.` : '')

function trimArgs(m: AgentMessage, call: ToolCall, w: Wording): ToolCall {
  const args: Record<string, unknown> = {}
  for (const [k, v] of Object.entries((call.args ?? {}) as Record<string, unknown>)) {
    args[k] = typeof v === 'string' && v.length > 200 ? `(${v.length} characters, left out.${back(w, m.id)})` : v
  }
  return { ...call, args }
}

function stub(m: AgentMessage, w: Wording): string {
  const lines = m.content ? m.content.split('\n').length : 0
  const first = (squeeze(m.content).split('\n').find((l) => l.trim()) ?? '').trim().slice(0, 100)
  const failed = m.toolResult && !m.toolResult.ok
  const size = lines > 1 ? `${lines} lines` : `${m.content.length} characters`
  const pictures = m.toolImages ? `, ${m.toolImages} picture${m.toolImages === 1 ? '' : 's'}` : ''
  return `[${failed ? 'failed' : 'result'} left out: ${size}${pictures}${first ? `, starting "${first}"` : ''}.${back(w, m.id)}]`
}

/** What marks one call as a repeat of another: the same tool with the same arguments. */
function repeatKey(m: AgentMessage): string | null {
  const call = m.toolCalls?.[0]
  if (!call || USER_SPEAKS.has(call.name)) return null
  return `${call.name}\u0000${JSON.stringify(call.args ?? {})}`
}

// ------------------------------------------------------------------ the view

export interface QuickView {
  messages: AgentMessage[]
  /** Messages whose attachments, images and attached documents are not sent. */
  dropSent: Set<string>
  /**
   * The note that the start is shortened, and the message it opens. It is already in that
   * message's text; a caller that sends a recorded copy of the message instead puts it in front of
   * that too (see withPreface), or the note would be lost exactly where the copy is used.
   */
  preface?: { id: string; text: string }
}

/** What a message is sent as, with the view's opening note in front when it is the message that carries it. */
export function withPreface<T extends { type: string; text?: string }>(view: QuickView, id: string, sent: string | T[] | undefined): string | T[] | undefined {
  if (sent === undefined || view.preface?.id !== id) return sent
  return typeof sent === 'string' ? `${view.preface.text}${BREAK}${sent}` : [{ type: 'text', text: view.preface.text } as T, ...sent]
}

const BREAK = '\n\n'

export interface ViewOptions {
  /** How many leading messages a summary already stands in for; they are not part of the view. */
  skip?: number
  /** Whether the model has the recall tools. Without them the notes do not mention them. */
  recall?: boolean
}

const list = (items: string[], max: number): string =>
  items.length <= max ? items.join('; ') : `${items.slice(0, max).join('; ')} and ${items.length - max} more`

/** An account of a turn's steps: what changed, what was looked at, what failed, what the user said, what was remarked. */
function ledger(steps: AgentMessage[], finalId: string | null, turn: number, w: Wording): string {
  const tools = steps.filter((m) => m.role === 'tool' && m.toolCalls?.[0])
  const remarks = steps.filter((m) => m.role === 'assistant' && m.id !== finalId && m.content.trim())
  if (!tools.length && !remarks.length) return ''
  const changed: string[] = []
  const ran: string[] = []
  const looked: string[] = []
  const failed: string[] = []
  const said: string[] = []
  const byName = new Map<string, number>()
  for (const m of tools) {
    const call = m.toolCalls![0]
    if (USER_SPEAKS.has(call.name)) {
      // The user's own words, kept in full: nothing in a session outranks them.
      said.push(`asked "${oneLine(callDetail(call), 300)}", the user answered "${oneLine(m.content, 600)}"`)
      continue
    }
    byName.set(call.name, (byName.get(call.name) ?? 0) + 1)
    const detail = callDetail(call)
    const into = isChange(call.name) ? changed : /^(run_|exec)/.test(call.name) ? ran : looked
    if (detail && !into.includes(detail)) into.push(detail)
    if (m.toolResult && !m.toolResult.ok) {
      failed.push(`${call.name}${detail ? ` ${detail}` : ''}: ${oneLine(squeeze(m.content).split('\n').find((l) => l.trim()) ?? 'failed', 110)}`)
    }
  }
  const parts: string[] = []
  const count = [...byName.values()].reduce((a, n) => a + n, 0)
  if (count) {
    parts.push(`${count} tool call${count === 1 ? '' : 's'} (${[...byName.entries()].map(([n, c]) => (c > 1 ? `${n} ×${c}` : n)).join(', ')})`)
  }
  if (changed.length) parts.push(`changed: ${list(changed, 14)}`)
  if (ran.length) parts.push(`ran: ${list(ran, 6)}`)
  if (looked.length) parts.push(`looked at: ${list(looked, 8)}`)
  if (failed.length) parts.push(`failed: ${list(failed, 4)}`)
  for (const s of said) parts.push(s)
  if (remarks.length) parts.push(`remarks on the way: ${list(remarks.map((m) => `"${oneLine(sentences(m.content, 1), 170)}"`), 8)}`)
  const how = w.recall ? ` Full text: recall_search with from_turn ${turn} and to_turn ${turn}.` : ''
  return `[Turn ${turn}, shortened. ${parts.join('. ')}.${how}]`
}

function collapse(region: AgentMessage[], level: number, open: boolean, turnOf: (m: AgentMessage) => number, w: Wording): AgentMessage[] {
  const out: AgentMessage[] = []
  let i = 0
  while (i < region.length) {
    const user = region[i].role === 'user' ? region[i] : null
    let j = user ? i + 1 : i
    while (j < region.length && region[j].role !== 'user') j++
    const steps = region.slice(user ? i + 1 : i, j)
    // A turn that carries on past this stretch has no final answer yet: everything said is a remark.
    const unfinished = open && j === region.length
    const last = unfinished ? undefined : [...steps].reverse().find((m) => m.role === 'assistant' && m.content.trim())
    const base = user ?? steps[0]
    const note = ledger(steps, last?.id ?? null, turnOf(base), w)
    const ask = user ? (level >= 5 ? headTail(user.content, 1500) : user.content) : ''
    const plan = user?.plan ? `[The plan for this turn]\n${headTail(user.plan, 1500)}` : ''
    const content = [ask, plan, note].filter(Boolean).join('\n\n')
    if (content) {
      out.push({ id: user ? user.id : `${base.id}:steps`, role: 'user', content, createdAt: base.createdAt })
    }
    if (last) {
      // A turn of many steps keeps its answer whole at strength 4: the steps are where the room was.
      const text = level >= 5 ? outline(last.content) : steps.length > 1 ? last.content : headTail(last.content, 2400)
      out.push({ id: last.id, role: 'assistant', content: text, createdAt: last.createdAt })
    } else if (!unfinished && user) {
      out.push({ id: `${user.id}:end`, role: 'assistant', content: '(This turn ended without a written answer.)', createdAt: user.createdAt })
    }
    i = j
  }
  return out
}

function shape(region: AgentMessage[], level: number, w: Wording): AgentMessage[] {
  // Reads and commands a later identical one made stale, found from the end backwards.
  const stale = new Set<string>()
  if (level >= 2) {
    const seen = new Set<string>()
    for (let i = region.length - 1; i >= 0; i--) {
      const key = region[i].role === 'tool' ? repeatKey(region[i]) : null
      if (!key) continue
      if (seen.has(key)) stale.add(region[i].id)
      seen.add(key)
    }
  }
  return region.map((m): AgentMessage => {
    if (m.role === 'tool') {
      const call = m.toolCalls?.[0]
      if (call && USER_SPEAKS.has(call.name)) return m
      const content =
        level >= 3 ? stub(m, w) : stale.has(m.id) ? `[Left out: the same call was made again later.${back(w, m.id)}]` : squeeze(m.content)
      return {
        ...m,
        content,
        toolCalls: level >= 3 ? m.toolCalls?.map((c) => trimArgs(m, c, w)) : m.toolCalls,
        toolResult: m.toolResult ? { ...m.toolResult, content } : m.toolResult,
        toolImages: level >= 3 ? undefined : m.toolImages
      }
    }
    if (m.role === 'assistant') return { ...m, reasoning: undefined }
    if (level < 3) return m
    const notes = [
      m.plan ? `[The plan for this turn]\n${headTail(m.plan, 1500)}` : '',
      m.attachments?.length ? `[Attachments not shown again.${w.recall ? ` recall_read("${m.id}") has what they said, and its pictures with images set.` : ''}]` : ''
    ].filter(Boolean)
    return notes.length ? { ...m, content: [m.content, ...notes].join('\n\n') } : m
  })
}

/** A spec's stretches that still stand: each ends at a message that exists, in order, after the summary. */
export function standing(all: AgentMessage[], spec: QuickSpec | null | undefined, skip = 0): { end: number; level: number }[] {
  const out: { end: number; level: number }[] = []
  let from = skip
  for (const seg of spec?.segments ?? []) {
    const end = all.findIndex((m) => m.id === seg.upto)
    // Rewound or deleted: this stretch and any after it no longer describe the conversation.
    if (end === -1) break
    // Wholly behind the summary: nothing of it is sent.
    if (end < from) continue
    out.push({ end, level: Math.max(1, Math.min(QUICK_LEVELS, Math.round(seg.level))) })
    from = end + 1
  }
  return out
}

/**
 * The messages to send for a conversation: its compacted stretches as the spec says, the rest as
 * they are. `all` is the whole conversation, so turns are numbered as recall numbers them.
 */
export function quickView(all: AgentMessage[], spec: QuickSpec | null | undefined, opts: ViewOptions = {}): QuickView {
  const skip = opts.skip ?? 0
  const dropSent = new Set<string>()
  const segments = standing(all, spec, skip)
  if (!segments.length) return { messages: skip ? all.slice(skip) : all, dropSent }
  const w: Wording = { recall: opts.recall !== false }

  const turns = new Map<string, number>()
  let n = 0
  for (const m of all) {
    if (m.role === 'user') n++
    turns.set(m.id, Math.max(1, n))
  }
  const turnOf = (m: AgentMessage): number => turns.get(m.id) ?? 1

  const out: AgentMessage[] = []
  const made = new Set<string>()
  let pos = skip
  for (const seg of segments) {
    const region = all.slice(pos, seg.end + 1)
    if (seg.level >= 3) for (const m of region) dropSent.add(m.id)
    const next = all[seg.end + 1]
    const rendered = seg.level >= 4 ? collapse(region, seg.level, !!next && next.role !== 'user', turnOf, w) : shape(region, seg.level, w)
    for (const m of rendered) {
      const prev = out.at(-1)
      // Two accounts in a row (a turn compacted in two stretches) read as one message.
      if (seg.level >= 4 && m.role === 'user' && prev?.role === 'user' && made.has(prev.id)) {
        out[out.length - 1] = { ...prev, content: `${prev.content}\n\n${m.content}` }
        continue
      }
      out.push(m)
      if (seg.level >= 4 && m.role === 'user') made.add(m.id)
    }
    pos = seg.end + 1
  }
  let preface: QuickView['preface']
  if (w.recall && out[0]?.role === 'user') {
    const text =
      '[The earlier part of this conversation is shown in a shortened form to save room. Nothing was deleted: ' +
      'recall_search finds anything in it and recall_read shows it in full. Turns are numbered from 1.]'
    preface = { id: out[0].id, text }
    out[0] = { ...out[0], content: `${text}${BREAK}${out[0].content}` }
  }
  return { messages: [...out, ...all.slice(pos)], dropSent, preface }
}

// ------------------------------------------------------------------ planning

export const LEVEL_NAMES = ['', 'Tidy', 'Drop repeats', 'Shorten tool results', 'Collapse turns', 'Outline']
export const LEVEL_DETAILS = [
  '',
  'Thinking dropped, tool output tidied. Nothing of substance is left out.',
  'Also drops reads and commands that were repeated later.',
  'Every tool result becomes one line; attachments and pictures are not sent again.',
  'Each turn becomes your message, an account of what followed, and the answer.',
  'Answers become outlines; the hardest, for when room matters most.'
]

export interface QuickOption {
  level: number
  name: string
  detail: string
  afterTokens: number
  spec: QuickSpec
}

export interface QuickPlan {
  ok: boolean
  /** Why not, when not. */
  message: string
  beforeTokens: number
  /** What no strength can go below: the instructions, the tools and the messages kept word for word. */
  floorTokens: number
  targetTokens: number
  windowTokens: number
  /** How many messages would be newly compacted. */
  messages: number
  /** One per strength, gentlest first. */
  options: QuickOption[]
  /** The gentlest strength that reaches the target, or the hardest when none does. */
  recommended: number
}

/**
 * What a quick compact would do at each strength, without doing it.
 *
 * `all` is the whole conversation; `fixedTokens` is what is sent whatever happens (instructions,
 * tools, any summary). Stretches already compacted are left exactly as they are: only what came
 * after them, up to the most recent messages, is newly shortened.
 */
export function planQuick(
  all: AgentMessage[],
  windowTokens: number,
  fixedTokens: number,
  opts: ViewOptions & { sentText?: (id: string) => string | undefined; previous?: QuickSpec | null } = {}
): QuickPlan {
  const skip = opts.skip ?? 0
  const cost = (view: QuickView): number =>
    view.messages.reduce((n, m) => n + costOf(m, (id) => (view.dropSent.has(id) ? undefined : opts.sentText?.(id))), 0)
  const kept = standing(all, opts.previous, skip)
  const prior: QuickSegment[] = kept.map((s) => ({ upto: all[s.end].id, level: s.level }))
  const before = fixedTokens + cost(quickView(all, { v: 2, segments: prior }, opts))
  const target = Math.floor(windowTokens * QUICK_TARGET_SHARE)
  const base: QuickPlan = { ok: false, message: '', beforeTokens: before, floorTokens: before, targetTokens: target, windowTokens, messages: 0, options: [], recommended: 0 }

  // The most recent messages stay as they are: back from the end until the keep is full.
  const keep = Math.min(QUICK_KEEP_MAX_TOKENS, Math.max(1000, Math.floor(windowTokens * QUICK_KEEP_SHARE)))
  const start = kept.length ? kept[kept.length - 1].end + 1 : skip
  let cut = all.length
  let keptTokens = 0
  while (cut > start) {
    const c = costOf(all[cut - 1], opts.sentText)
    // The last two messages are kept whatever they cost.
    if (keptTokens + c > keep && all.length - cut >= 2) break
    keptTokens += c
    cut--
  }
  if (cut <= start) {
    return { ...base, message: kept.length ? 'Nothing new to compact: everything but the most recent messages already is.' : 'Nothing to compact yet: the conversation is only its most recent messages.' }
  }

  // Where the newly compacted stretch divides into an older and a newer half, by size, at the start of a turn.
  const sizes = all.slice(start, cut).map((m) => costOf(m, opts.sentText))
  const half = sizes.reduce((a, n) => a + n, 0) / 2
  let split = -1
  let run = 0
  for (let i = start; i < cut - 1; i++) {
    run += sizes[i - start]
    if (run >= half && all[i + 1].role === 'user') {
      split = i
      break
    }
  }

  const options: QuickOption[] = []
  for (let level = 1; level <= QUICK_LEVELS; level++) {
    const added: QuickSegment[] =
      level >= 4 && split !== -1
        ? [
            { upto: all[split].id, level },
            // The newer half, one strength gentler: what is nearer is more likely to matter.
            { upto: all[cut - 1].id, level: level - 1 }
          ]
        : [{ upto: all[cut - 1].id, level }]
    const spec: QuickSpec = { v: 2, segments: [...prior, ...added] }
    options.push({ level, name: LEVEL_NAMES[level], detail: LEVEL_DETAILS[level], afterTokens: fixedTokens + cost(quickView(all, spec, opts)), spec })
  }
  const floor = fixedTokens + keptTokens
  const best = options[options.length - 1].afterTokens
  const plan = { ...base, floorTokens: floor, messages: cut - start, options }
  if (before - best < Math.max(300, before * 0.05)) {
    const why =
      floor > before * 0.85
        ? `the instructions, tools and most recent messages alone are about ${floor.toLocaleString()} tokens, which no compaction shrinks`
        : 'it is already about as small as this can make it'
    return { ...plan, message: `A quick compact would free very little here: ${why}.` }
  }
  // Only strengths that free something are offered: on a chat with no tool output the gentle ones change nothing.
  const worth = options.filter((o) => before - o.afterTokens >= Math.max(100, before * 0.02))
  const recommended = (worth.find((o) => o.afterTokens <= target) ?? worth[worth.length - 1]).level
  return { ...plan, options: worth, ok: true, message: '', recommended }
}
