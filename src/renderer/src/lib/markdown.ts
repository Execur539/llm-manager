/**
 * Markdown parsing, with no rendering and no React.
 *
 * Kept separate from the component so the grammar can be unit-tested directly: list nesting,
 * ordered-versus-unordered boundaries, unterminated code fences and inline precedence are all
 * easy to get subtly wrong, and checking them through rendered output is slow and indirect.
 */

export type Inline = { type: 'text' | 'code' | 'bold' | 'italic' | 'strike' | 'link'; value: string; href?: string }

/**
 * Tokenise a single line of inline markup.
 *
 * Code spans bind tightest and are captured first, so `**not bold**` inside backticks stays
 * literal — which matters when the model is explaining markdown itself.
 */
export function parseInline(text: string): Inline[] {
  const tokens: Inline[] = []
  let buffer = ''

  const flush = (): void => {
    if (buffer) {
      tokens.push({ type: 'text', value: buffer })
      buffer = ''
    }
  }

  for (let i = 0; i < text.length; ) {
    const rest = text.slice(i)

    const code = rest.match(/^`([^`]+)`/)
    if (code) {
      flush()
      tokens.push({ type: 'code', value: code[1] })
      i += code[0].length
      continue
    }

    const link = rest.match(/^\[([^\]]*)\]\(([^)\s]+)[^)]*\)/)
    if (link) {
      flush()
      tokens.push({ type: 'link', value: link[1] || link[2], href: link[2] })
      i += link[0].length
      continue
    }

    const bold = rest.match(/^(\*\*|__)(.+?)\1/)
    if (bold) {
      flush()
      tokens.push({ type: 'bold', value: bold[2] })
      i += bold[0].length
      continue
    }

    const strike = rest.match(/^~~(.+?)~~/)
    if (strike) {
      flush()
      tokens.push({ type: 'strike', value: strike[1] })
      i += strike[0].length
      continue
    }

    // Single asterisk/underscore, but not mid-word (snake_case must survive).
    const italic = rest.match(/^(\*|_)(?!\s)(.+?)(?<!\s)\1/)
    if (italic && !(italic[1] === '_' && /\w/.test(text[i - 1] ?? ''))) {
      flush()
      tokens.push({ type: 'italic', value: italic[2] })
      i += italic[0].length
      continue
    }

    buffer += text[i]
    i++
  }
  flush()
  return tokens
}

// ---------------------------------------------------------------- blocks

export type Block =
  | { kind: 'p'; lines: string[] }
  | { kind: 'heading'; level: number; text: string }
  | { kind: 'code'; lang: string; lines: string[] }
  | { kind: 'list'; ordered: boolean; items: { text: string; depth: number }[] }
  | { kind: 'quote'; lines: string[] }
  | { kind: 'table'; header: string[]; rows: string[][] }
  | { kind: 'hr' }

const splitRow = (line: string): string[] =>
  line
    .replace(/^\s*\|/, '')
    .replace(/\|\s*$/, '')
    .split('|')
    .map((c) => c.trim())

/** A block and the lines it was read from: `start` inclusive, `end` exclusive. */
interface BlockSpan {
  block: Block
  start: number
  end: number
}

function parseSpans(lines: string[]): BlockSpan[] {
  const spans: BlockSpan[] = []
  let i = 0
  // Always called after `i` has moved past the block, so `end` is where the next one may begin.
  const push = (block: Block, start: number): void => {
    spans.push({ block, start, end: Math.min(i, lines.length) })
  }

  while (i < lines.length) {
    const line = lines[i]

    if (!line.trim()) {
      i++
      continue
    }
    const start = i

    // Fenced code. An unterminated fence runs to the end, which is what a streaming reply
    // looks like mid-block — it should render as code, not as prose.
    const fence = line.match(/^\s*```(\w*)/)
    if (fence) {
      const lang = fence[1]
      const body: string[] = []
      i++
      while (i < lines.length && !/^\s*```/.test(lines[i])) {
        body.push(lines[i])
        i++
      }
      i++
      push({ kind: 'code', lang, lines: body }, start)
      continue
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/)
    if (heading) {
      i++
      push({ kind: 'heading', level: heading[1].length, text: heading[2].trim() }, start)
      continue
    }

    if (/^\s*(---+|\*\*\*+|___+)\s*$/.test(line)) {
      i++
      push({ kind: 'hr' }, start)
      continue
    }

    // Table: a pipe row followed by a separator row.
    if (line.includes('|') && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[i + 1])) {
      const header = splitRow(line)
      i += 2
      const rows: string[][] = []
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
        rows.push(splitRow(lines[i]))
        i++
      }
      push({ kind: 'table', header, rows }, start)
      continue
    }

    if (/^\s*>/.test(line)) {
      const body: string[] = []
      while (i < lines.length && /^\s*>/.test(lines[i])) {
        body.push(lines[i].replace(/^\s*>\s?/, ''))
        i++
      }
      push({ kind: 'quote', lines: body }, start)
      continue
    }

    const listStart = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/)
    if (listStart) {
      const ordered = /\d/.test(listStart[2])
      const items: { text: string; depth: number }[] = []
      while (i < lines.length) {
        const m = lines[i].match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/)
        if (!m) {
          // A plain indented line continues the previous item.
          if (items.length && /^\s+\S/.test(lines[i]) && lines[i].trim()) {
            items[items.length - 1].text += ` ${lines[i].trim()}`
            i++
            continue
          }
          break
        }
        // A change of marker type at the top level starts a new list. Without this, "1. a"
        // following "- b" is absorbed into the bulleted list and loses its numbering.
        if (Math.floor(m[1].length / 2) === 0 && /\d/.test(m[2]) !== ordered) break
        items.push({ text: m[3], depth: Math.floor(m[1].length / 2) })
        i++
      }
      push({ kind: 'list', ordered, items }, start)
      continue
    }

    const para: string[] = []
    while (i < lines.length && lines[i].trim() && !/^\s*(#{1,6}\s|```|>|---+$)/.test(lines[i])) {
      const isListItem = /^(\s*)([-*+]|\d+[.)])\s+/.test(lines[i])
      if (isListItem && para.length) break
      para.push(lines[i])
      i++
      if (isListItem) break
    }
    if (para.length) push({ kind: 'p', lines: para }, start)
    else i++
  }

  return spans
}

export function parseBlocks(source: string): Block[] {
  return parseSpans(source.replace(/\r\n/g, '\n').split('\n')).map((s) => s.block)
}

/**
 * How many of `spans` can no longer change: the index of the first one that still can.
 *
 * Only ever a point where lines lie between two blocks. The parser reads forwards and looks at most
 * one line ahead, and a line with another after it is finished — so once such a gap exists the
 * blocks before it are fixed, and parsing the text on each side of it separately gives exactly the
 * blocks that parsing the whole would. The last `keepLive` blocks stay live regardless, because a
 * block still being written can still turn into a different kind of block.
 */
function settledCount(spans: BlockSpan[], keepLive: number): number {
  for (let k = spans.length - keepLive; k > 0; k--) {
    if (spans[k].start > spans[k - 1].end) return k
  }
  return 0
}

/** Split a streaming reply into the part that can no longer change and the part that still can. */
export function splitSettled(source: string, keepLive = 2): { settled: string; tail: string } {
  const text = source.replace(/\r\n/g, '\n')
  const lines = text.split('\n')
  const spans = parseSpans(lines)
  const k = settledCount(spans, keepLive)
  if (!k) return { settled: '', tail: text }
  const cut = spans[k].start
  return { settled: `${lines.slice(0, cut).join('\n')}\n`, tail: lines.slice(cut).join('\n') }
}

/** A streaming reply's blocks: those that can no longer change, then those that still can. */
export interface StreamingBlocks {
  settled: Block[]
  live: Block[]
}

/**
 * Parse a reply as it streams, doing work in proportion to what changed rather than to its length.
 *
 * Parsing the whole reply for every token made a long answer quadratic: by the end of it each new
 * word re-read everything before it. This remembers the text that has settled and the blocks it
 * gave, and each call parses only what follows that text. Blocks that settle are appended, and the
 * ones already settled are handed back as the same objects — so a view memoised on them skips all
 * but the blocks still being written. A source that no longer starts with the settled text, such as
 * a different reply, starts over.
 */
export function streamingParser(keepLive = 2): (source: string) => StreamingBlocks {
  let settledText = ''
  let settled: Block[] = []
  return (raw) => {
    const source = raw.replace(/\r\n/g, '\n')
    if (!source.startsWith(settledText)) {
      settledText = ''
      settled = []
    }
    const lines = source.slice(settledText.length).split('\n')
    const spans = parseSpans(lines)
    const k = settledCount(spans, keepLive)
    if (k) {
      settled = settled.concat(spans.slice(0, k).map((s) => s.block))
      settledText += `${lines.slice(0, spans[k].start).join('\n')}\n`
    }
    return { settled, live: spans.slice(k).map((s) => s.block) }
  }
}
