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

export function parseBlocks(source: string): Block[] {
  const lines = source.replace(/\r\n/g, '\n').split('\n')
  const blocks: Block[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]

    if (!line.trim()) {
      i++
      continue
    }

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
      blocks.push({ kind: 'code', lang, lines: body })
      continue
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/)
    if (heading) {
      blocks.push({ kind: 'heading', level: heading[1].length, text: heading[2].trim() })
      i++
      continue
    }

    if (/^\s*(---+|\*\*\*+|___+)\s*$/.test(line)) {
      blocks.push({ kind: 'hr' })
      i++
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
      blocks.push({ kind: 'table', header, rows })
      continue
    }

    if (/^\s*>/.test(line)) {
      const body: string[] = []
      while (i < lines.length && /^\s*>/.test(lines[i])) {
        body.push(lines[i].replace(/^\s*>\s?/, ''))
        i++
      }
      blocks.push({ kind: 'quote', lines: body })
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
      blocks.push({ kind: 'list', ordered, items })
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
    if (para.length) blocks.push({ kind: 'p', lines: para })
    else i++
  }

  return blocks
}
