/**
 * Recall: the model looking back into its own conversation.
 *
 * Compaction leaves things out of what the model is sent, but the transcript keeps every word.
 * These two tools reach it, in two steps so that looking is cheap: `recall_search` returns a short
 * list of where something is (a line or two per hit), and `recall_read` returns one of them, and
 * only the part asked for. A model hunting for a detail therefore spends a few hundred tokens
 * finding it rather than refilling its window with whole tool results on the way.
 *
 * Search is by words, by regular expression or by meaning, narrowed by who said it, which tool it
 * came from, what file it touched, which turns, and whether it failed; with no query it lists the
 * turns, or what one turn holds. A tool call is searched and read with its arguments, so the text
 * of an edit or a written file is as reachable as a result. What is still in front of the model
 * word for word is not offered again.
 *
 * Words are matched the way a person searches: filler words are ignored, endings do not matter,
 * rare words count for more than common ones, and what was said counts for more than a tool's
 * output. When no message has every word, the closest are given and said to be only close, so
 * that "nothing matches" is never mistaken for "it never happened".
 *
 * Pure apart from the functions it is handed, so it is checked from a script.
 */

import vm from 'node:vm'
import type { AgentMessage, ToolDefinition } from '@shared/types'
import type { ContentPart } from '../runtime/llama'
import { callDetail } from './quick-compact'

// The same shapes agent/tools/base builds; written out here so this file imports nothing of the app.
const schema = (properties: Record<string, unknown>, required: string[] = []): Record<string, unknown> => ({ type: 'object', properties, required })
const str = (description: string): Record<string, unknown> => ({ type: 'string', description })
const int = (description: string): Record<string, unknown> => ({ type: 'integer', description })
const bool = (description: string): Record<string, unknown> => ({ type: 'boolean', description })

export const RECALL_TOOLS: ToolDefinition[] = [
  {
    name: 'recall_search',
    tier: 'read',
    description:
      'Look back into the earlier part of this conversation that is no longer shown to you in full (it was shortened ' +
      'to save room; every word is kept). Returns a short list, never the text itself: each match has an id, its ' +
      'turn, who or which tool it came from, and the matching line. Then read the one you need with recall_read. ' +
      'Search by words (default: endings and filler words do not matter), by regular expression, or by meaning when ' +
      'you do not know the wording. Narrow with role, tool, path, turns or errors_only rather than reading many ' +
      'results. With no query it lists the turns; with a turn and no query, everything in that turn.',
    parameters: schema({
      query: str('What to look for: words, a regular expression, or a description of what it was about. Leave out to list turns.'),
      mode: str("'words' (default), 'regex', or 'meaning' (slower the first time in a long conversation)"),
      role: str("Only messages from 'user', 'assistant' or 'tool'"),
      tool: str('Only calls of this tool, e.g. read_file or run_command'),
      path: str('Only tool calls whose file, command or address contains this text'),
      from_turn: int("Only from this turn on (turns are numbered from 1 by the user's messages)"),
      to_turn: int('Only up to this turn'),
      errors_only: bool('Only tool calls that failed'),
      context: int('Lines to show either side of each matching line (0 to 3, default 0)'),
      limit: int('How many to return (default 8, at most 20)'),
      offset: int('Skip this many, to see the next page')
    })
  },
  {
    name: 'recall_read',
    tier: 'read',
    description:
      'Read one earlier message or tool call in full, by the id recall_search gave or the id in a "left out" note. A ' +
      "tool call is shown with its arguments, then its result. Long ones are read a part at a time: give around (text " +
      'to centre on) or start_line and end_line. Set images to see the pictures it had. Read only what you need: what ' +
      'you read stays in your context.',
    parameters: schema(
      {
        id: str('The id of the message or tool call'),
        around: str('Show the lines around the first place this text appears'),
        start_line: int('First line to show (lines are numbered from 1)'),
        end_line: int('Last line to show'),
        images: bool('Also show the pictures it was sent with or returned')
      },
      ['id']
    )
  }
]

export const isRecallTool = (name: string): boolean => name === 'recall_search' || name === 'recall_read'

/** The most one call returns, so a look back never costs much of the window. */
const READ_MAX_CHARS = 6000
const SNIPPET_CHARS = 220
/** How many pieces one search by meaning will work out afresh; the rest wait for the next. */
const EMBED_PER_CALL = 400

interface Unit {
  message: AgentMessage
  turn: number
  /** Position in the conversation. */
  index: number
  tool?: string
  detail: string
  text: string
  low: string
}

/** A tool call as text: what it was called with, then what came back. */
function callText(m: AgentMessage): string {
  const call = m.toolCalls?.[0]
  if (!call) return m.content
  const args = Object.entries((call.args ?? {}) as Record<string, unknown>).map(([k, v]) =>
    typeof v === 'string' ? (v.includes('\n') ? `${k}:\n${v}` : `${k}: ${v}`) : `${k}: ${JSON.stringify(v)}`
  )
  return [`[call] ${call.name}`, ...args, '[result]', m.content].join('\n')
}

function unitsOf(messages: AgentMessage[], sentText?: (id: string) => string | undefined): Unit[] {
  let turn = 0
  return messages.map((message, index) => {
    if (message.role === 'user') turn++
    const call = message.role === 'tool' ? message.toolCalls?.[0] : undefined
    const body = message.role === 'tool' ? callText(message) : ((message.role === 'user' ? sentText?.(message.id) : undefined) ?? message.content)
    const text = message.role === 'assistant' && message.reasoning ? `${body}\n[thinking]\n${message.reasoning}` : body
    return { message, turn: Math.max(1, turn), index, tool: call?.name, detail: callDetail(call), text, low: text.toLowerCase() }
  })
}

const label = (u: Unit): string =>
  u.message.role === 'tool' ? `tool ${u.tool ?? '?'}${u.detail ? ` ${u.detail}` : ''}${u.message.toolResult && !u.message.toolResult.ok ? ' (failed)' : ''}` : u.message.role

function clip(line: string, at = 0): string {
  const t = line.trim()
  if (t.length <= SNIPPET_CHARS) return t
  const start = Math.max(0, Math.min(at - 60, t.length - SNIPPET_CHARS))
  return `${start > 0 ? '…' : ''}${t.slice(start, start + SNIPPET_CHARS)}${start + SNIPPET_CHARS < t.length ? '…' : ''}`
}

export interface RecallOptions {
  /**
   * Messages from this position on are still sent to the model word for word, so they are never
   * offered as results. Zero when nothing has been compacted.
   */
  visibleFrom: number
  sentText?: (id: string) => string | undefined
  /** The pictures a message was sent with or returned, for recall_read with images set. */
  sentParts?: (id: string) => ContentPart[] | undefined
  /** Turns texts into vectors, for search by meaning. Absent when no embedding model can run. */
  embed?: (texts: string[]) => Promise<number[][]>
  /** Vectors already worked out for this conversation, by message id; kept by the caller. */
  vectors?: Map<string, Float32Array[]>
  signal?: AbortSignal
}

interface Hit {
  unit: Unit
  score: number
  line: number
  snippet: string
  /** Matching lines beyond the one shown. */
  more: number
}

// ------------------------------------------------------------------ words

const FILLER = new Set(
  ('a an the and or but if then so of to in on at by for from with without about into over under is are was were be been ' +
    'being do does did done have has had having i me my we us our you your it its this that these those there here ' +
    'what which who whom whose when where why how can could should would will shall may might must not no yes as than ' +
    'too very just also any some all each every both either neither please tell show find look get got give use used').split(' ')
)

/** A word without its ending, so "compacted" finds "compaction". Short words are left alone. */
function stem(word: string): string {
  if (word.length < 5) return word
  // "dropping" and "drops" both come down to "drop".
  const cut = word.replace(/(ations?|ions?|ings?|ments?|ed|es|ly|s)$/, '') || word
  return cut.length > 3 && cut.at(-1) === cut.at(-2) ? cut.slice(0, -1) : cut
}

function termsOf(query: string): string[] {
  const words = query.toLowerCase().match(/[\p{L}\p{N}_][\p{L}\p{N}_.\-/\\]*/gu) ?? []
  const kept = words.map((w) => w.replace(/[.\-/\\]+$/, '')).filter((w) => w && !FILLER.has(w))
  // A query of nothing but filler words is searched as written.
  return [...new Set((kept.length ? kept : words).map(stem))]
}

function byWords(units: Unit[], query: string, context: number): { hits: Hit[]; loose: boolean } {
  const terms = termsOf(query)
  if (!terms.length) return { hits: [], loose: false }
  const phrase = query.toLowerCase().trim()
  const has = units.map((u) => terms.map((t) => u.low.includes(t)))
  // Rare words tell more than common ones.
  const weight = terms.map((_, k) => Math.log(1 + units.length / (1 + has.filter((h) => h[k]).length)))
  const everyWord = has.some((h) => h.every(Boolean))
  const hits: Hit[] = []
  units.forEach((u, i) => {
    const present = has[i]
    if (everyWord ? !present.every(Boolean) : !present.some(Boolean)) return
    const lines = u.text.split('\n')
    let best = { n: 0, line: 0, at: 0 }
    let matching = 0
    lines.forEach((raw, li) => {
      const low = raw.toLowerCase()
      let n = 0
      let at = -1
      terms.forEach((t, k) => {
        const p = low.indexOf(t)
        if (p === -1) return
        n += weight[k]
        if (at === -1) at = p
      })
      if (phrase.length > 3 && low.includes(phrase)) n += 2
      if (n > 0) matching++
      if (n > best.n) best = { n, line: li, at: Math.max(0, at) }
    })
    if (!best.n) return
    const found = present.reduce((a, p, k) => a + (p ? weight[k] : 0), 0)
    // What was said outranks a tool's output: a decision is two lines, a file read is thousands.
    const voice = u.message.role === 'tool' ? 1 : 1.5
    hits.push({ unit: u, score: (found + best.n) * voice, line: best.line + 1, snippet: snippetAt(lines, best.line, best.at, context), more: matching - 1 })
  })
  return { hits, loose: !everyWord && terms.length > 1 }
}

function snippetAt(lines: string[], line: number, at: number, context: number): string {
  if (context <= 0) return clip(lines[line], at)
  const from = Math.max(0, line - context)
  const to = Math.min(lines.length - 1, line + context)
  const out: string[] = []
  for (let i = from; i <= to; i++) out.push(`\n    ${i + 1}${i === line ? '>' : ':'} ${clip(lines[i], i === line ? at : 0)}`)
  return out.join('')
}

// ------------------------------------------------------------------ regex

/**
 * Matched inside a sandbox with a time limit. The pattern is the model's, the text can be megabytes,
 * and a pattern that backtracks without end would otherwise hang the whole app.
 */
function byRegex(units: Unit[], pattern: string, context: number): Hit[] | string {
  let re: RegExp
  try {
    re = new RegExp(pattern, 'i')
  } catch (err) {
    return `That is not a valid regular expression: ${err instanceof Error ? err.message : String(err)}`
  }
  const box = vm.createContext({ re, lines: [] as string[], out: [] as number[] })
  const script = new vm.Script('out = lines.map((l) => { const m = re.exec(l.length > 4000 ? l.slice(0, 4000) : l); return m ? m.index : -1 })')
  const hits: Hit[] = []
  const deadline = Date.now() + 3000
  for (const u of units) {
    const lines = u.text.split('\n')
    box.lines = lines
    try {
      script.runInContext(box, { timeout: 250 })
    } catch {
      return 'That pattern takes too long to match against this conversation. Use a simpler one, or search by words.'
    }
    const at = box.out as number[]
    const first = at.findIndex((p) => p !== -1)
    if (first !== -1) {
      const n = at.filter((p) => p !== -1).length
      hits.push({ unit: u, score: 1 + Math.min(1, n / 50), line: first + 1, snippet: snippetAt(lines, first, at[first], context), more: n - 1 })
    }
    if (Date.now() > deadline) return 'That pattern takes too long to match against this conversation. Narrow it with turns, role or tool.'
  }
  return hits
}

// ------------------------------------------------------------------ meaning

function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  return na && nb ? dot / Math.sqrt(na * nb) : 0
}

/** A message as pieces small enough to embed, each remembering the line it starts on. */
function chunksOf(text: string): { text: string; line: number }[] {
  const out: { text: string; line: number }[] = []
  let buf: string[] = []
  let size = 0
  let start = 1
  text.split('\n').forEach((l, i) => {
    if (size + l.length > 1200 && buf.length) {
      out.push({ text: buf.join('\n'), line: start })
      buf = []
      size = 0
      start = i + 1
    }
    buf.push(l.length > 1200 ? l.slice(0, 1200) : l)
    size += l.length + 1
  })
  if (buf.join('').trim()) out.push({ text: buf.join('\n'), line: start })
  return out
}

/**
 * By meaning. Working out the vectors is the slow part, so each call does a bounded amount of it,
 * most recent first, keeps what it made, and says how much of the conversation it has covered: a
 * very long one is covered over a few searches rather than in one wait of minutes.
 */
async function byMeaning(units: Unit[], query: string, opts: RecallOptions): Promise<{ hits: Hit[]; note: string } | null> {
  if (!opts.embed) return null
  const cache = opts.vectors ?? new Map<string, Float32Array[]>()
  // The first two hundred pieces of anything: one enormous attached document must not be skipped for ever.
  const pieces = units.map((u) => ({ u, chunks: chunksOf(u.text).slice(0, 200) })).filter((p) => p.chunks.length)
  const todo = pieces.filter((p) => cache.get(p.u.message.id)?.length !== p.chunks.length).reverse()
  let budget = EMBED_PER_CALL
  let left = 0
  for (const p of todo) {
    if (opts.signal?.aborted) return null
    if (p.chunks.length > budget) {
      left++
      continue
    }
    budget -= p.chunks.length
    const made: Float32Array[] = []
    for (let i = 0; i < p.chunks.length; i += 32) {
      const vectors = await opts.embed(p.chunks.slice(i, i + 32).map((c) => c.text))
      for (const v of vectors) made.push(Float32Array.from(v))
    }
    cache.set(p.u.message.id, made)
  }
  const [q] = await opts.embed([query])
  if (!q) return null
  const hits: Hit[] = []
  for (const p of pieces) {
    const vectors = cache.get(p.u.message.id)
    if (!vectors || vectors.length !== p.chunks.length) continue
    let best = -1
    let at = 0
    vectors.forEach((v, k) => {
      const s = cosine(q, v)
      if (s > best) {
        best = s
        at = k
      }
    })
    if (best <= 0.2) continue
    const chunk = p.chunks[at]
    hits.push({ unit: p.u, score: best, line: chunk.line, snippet: clip(chunk.text.split('\n').find((l) => l.trim()) ?? ''), more: 0 })
  }
  const note = left
    ? `Searched by meaning across ${pieces.length - left} of ${pieces.length} earlier items so far; search again to cover the rest, or use words.\n`
    : ''
  return { hits, note }
}

// ------------------------------------------------------------------ search

/** Find where something is in the earlier conversation: a short list, never the text itself. */
export async function recallSearch(messages: AgentMessage[], args: Record<string, unknown>, opts: RecallOptions): Promise<string> {
  const query = String(args.query ?? '').trim()
  const mode = String(args.mode ?? 'words').toLowerCase()
  const role = typeof args.role === 'string' ? args.role.toLowerCase() : ''
  const tool = typeof args.tool === 'string' ? args.tool.toLowerCase() : ''
  const pathPart = typeof args.path === 'string' ? args.path.toLowerCase() : ''
  const given = (v: unknown): boolean => v != null && v !== '' && Number.isFinite(Number(v))
  const from = given(args.from_turn) ? Number(args.from_turn) : -Infinity
  const to = given(args.to_turn) ? Number(args.to_turn) : Infinity
  const errorsOnly = args.errors_only === true || args.errors_only === 'true'
  const context = Math.max(0, Math.min(3, Math.round(Number(args.context) || 0)))
  const limit = Math.max(1, Math.min(20, Math.round(Number(args.limit) || 8)))
  const offset = Math.max(0, Math.round(Number(args.offset) || 0))

  const all = unitsOf(messages, opts.sentText)
  if (opts.visibleFrom <= 0) return 'Nothing has been compacted: the whole conversation is still in front of you.'
  const earlier = all.filter((u) => u.index < opts.visibleFrom && !(u.tool && isRecallTool(u.tool)))
  const filtered = !!(role || tool || pathPart || errorsOnly || from !== -Infinity || to !== Infinity)
  const units = earlier.filter(
    (u) =>
      (!role || u.message.role === role) &&
      (!tool || (u.tool ?? '').toLowerCase() === tool) &&
      (!pathPart || u.detail.toLowerCase().includes(pathPart) || (u.message.role === 'tool' && u.low.split('[result]')[0].includes(pathPart))) &&
      u.turn >= from &&
      u.turn <= to &&
      (!errorsOnly || (u.message.role === 'tool' && !!u.message.toolResult && !u.message.toolResult.ok))
  )
  const page = <T>(list: T[]): { rows: T[]; tail: string } => {
    const rows = list.slice(offset, offset + limit)
    const left = list.length - offset - rows.length
    return { rows, tail: left > 0 ? `\n${left} more: call again with offset ${offset + rows.length}, or narrow it.` : '' }
  }

  // No query: a table of contents. The turns, or with filters, what they hold.
  if (!query) {
    if (!filtered) {
      const turns = new Map<number, Unit[]>()
      for (const u of earlier) turns.set(u.turn, [...(turns.get(u.turn) ?? []), u])
      const { rows, tail } = page([...turns.entries()])
      const lines = rows.map(([n, us]) => {
        const ask = us.find((u) => u.message.role === 'user')
        const calls = us.filter((u) => u.message.role === 'tool').length
        return `turn ${n}: ${ask ? `"${clip(ask.message.content.replace(/\s+/g, ' ')).slice(0, 110)}"` : '(continues an earlier request)'} · ${calls} tool call${calls === 1 ? '' : 's'}`
      })
      return `${turns.size} earlier turn${turns.size === 1 ? '' : 's'}:\n${lines.join('\n')}${tail}\nList one with from_turn and to_turn, or search with a query.`
    }
    if (!units.length) return 'Nothing earlier matches those filters.'
    const { rows, tail } = page(units)
    const lines = rows.map((u) => `[${u.message.id}] turn ${u.turn} · ${label(u)} · ${clip(u.message.content.split('\n').find((l) => l.trim()) ?? '')}`)
    return `${units.length} item${units.length === 1 ? '' : 's'}, in order:\n${lines.join('\n')}${tail}\nRead one with recall_read.`
  }
  if (!units.length) return 'Nothing earlier matches those filters.'

  let note = ''
  let hits: Hit[] | null = null
  if (mode === 'meaning') {
    const found = await byMeaning(units, query, opts).catch(() => null)
    if (found) {
      hits = found.hits
      note = found.note
    } else {
      note = 'Search by meaning is not available right now, so this is a search by words.\n'
    }
  }
  if (!hits && mode === 'regex') {
    const found = byRegex(units, query, context)
    if (typeof found === 'string') return found
    hits = found
  }
  if (!hits) {
    const found = byWords(units, query, context)
    hits = found.hits
    if (found.loose && hits.length) note += 'No single earlier item has all of those words. The closest:\n'
  }
  // Best first; among equals, the most recent.
  hits.sort((a, b) => b.score - a.score || b.unit.index - a.unit.index)
  const recent = all.length - opts.visibleFrom
  if (!hits.length) {
    return `${note}No match in the earlier conversation${filtered ? ' with those filters' : ''}.${recent > 0 ? ' The most recent messages are still in front of you and were not searched.' : ''} Try other words, search by meaning, or list the turns by leaving the query out.`
  }
  const { rows, tail } = page(hits)
  const lines = rows.map(
    (h) => `[${h.unit.message.id}] turn ${h.unit.turn} · ${label(h.unit)} · line ${h.line}${h.more > 0 ? ` (+${h.more} more lines)` : ''}: ${h.snippet}`
  )
  return (
    `${note}${hits.length} match${hits.length === 1 ? '' : 'es'}${offset ? `, from number ${offset + 1}` : ''}:\n` +
    lines.join('\n') +
    tail +
    '\nRead one with recall_read, giving its id and around or a line range.'
  )
}

// ------------------------------------------------------------------ read

/** One earlier message or tool call, or the asked-for part of it, with its lines numbered. */
export function recallRead(
  messages: AgentMessage[],
  args: Record<string, unknown>,
  opts: Pick<RecallOptions, 'sentText' | 'sentParts'>
): string | { text: string; media: ContentPart[] } {
  // An id as a note gave it, with or without brackets; a shortened turn's own id belongs to its first message.
  const id = String(args.id ?? '').trim().replace(/^\[|\]$/g, '').replace(/:(steps|end)$/, '')
  if (!id) return 'Give the id of what to read: the one recall_search listed, or the one in a "left out" note.'
  const unit = unitsOf(messages, opts.sentText).find((u) => u.message.id === id)
  if (!unit) return `Nothing in this conversation has the id ${id}. Find it with recall_search.`
  const lines = unit.text.split('\n')
  let start = 1
  let end = lines.length
  const around = typeof args.around === 'string' ? args.around.trim() : ''
  if (around) {
    const at = lines.findIndex((l) => l.toLowerCase().includes(around.toLowerCase()))
    if (at === -1) return `"${around}" does not appear in ${id} (${lines.length} lines). Leave around out to read from its start.`
    start = Math.max(1, at + 1 - 15)
    end = Math.min(lines.length, at + 1 + 25)
  } else if (args.start_line != null || args.end_line != null) {
    start = Math.max(1, Math.round(Number(args.start_line) || 1))
    end = Math.min(lines.length, Math.round(Number(args.end_line) || start + 80))
    if (start > lines.length) return `${id} has only ${lines.length} lines.`
  }
  const out: string[] = []
  let used = 0
  let last = start - 1
  for (let i = start; i <= end; i++) {
    const line = lines[i - 1].length > 500 ? `${lines[i - 1].slice(0, 480)}… (${lines[i - 1].length} characters)` : lines[i - 1]
    const row = `${i}: ${line}`
    if (used + row.length > READ_MAX_CHARS && out.length) break
    out.push(row)
    used += row.length + 1
    last = i
  }
  const head = `[${id}] turn ${unit.turn} · ${label(unit)} · lines ${start}-${last} of ${lines.length}`
  const text = `${head}\n${out.join('\n')}${last < lines.length ? `\nMore: recall_read with start_line ${last + 1}.` : ''}`
  if (args.images !== true && args.images !== 'true') return text
  const media = (opts.sentParts?.(id) ?? []).filter((p) => p.type === 'image_url')
  return media.length ? { text: `${text}\nIts ${media.length} picture${media.length === 1 ? '' : 's'} follow.`, media } : `${text}\nIt has no pictures that are still kept.`
}
