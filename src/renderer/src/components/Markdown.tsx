import { Fragment, type JSX, type ReactNode } from 'react'

/**
 * Minimal markdown renderer for assistant messages.
 *
 * Builds React elements directly rather than producing an HTML string. Model output is
 * untrusted — it can repeat anything it read from a file or a web page — so it must never reach
 * `dangerouslySetInnerHTML`. React escapes every text node here by construction, which removes
 * the injection surface entirely instead of trying to sanitise it away.
 *
 * Scope is deliberately what a chat reply actually uses: headings, emphasis, code, lists, tables,
 * quotes, links and rules. Anything unrecognised falls through as plain text rather than being
 * silently dropped.
 */

// ---------------------------------------------------------------- inline

type Inline = { type: 'text' | 'code' | 'bold' | 'italic' | 'strike' | 'link'; value: string; href?: string }

/**
 * Tokenise a single line of inline markup.
 *
 * Code spans bind tightest and are captured first, so `**not bold**` inside backticks stays
 * literal — which matters when the model is explaining markdown itself.
 */
function parseInline(text: string): Inline[] {
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

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  return parseInline(text).map((token, i) => {
    const key = `${keyPrefix}-${i}`
    switch (token.type) {
      case 'code':
        return (
          <code className="md-code" key={key}>
            {token.value}
          </code>
        )
      case 'bold':
        return <strong key={key}>{renderInline(token.value, key)}</strong>
      case 'italic':
        return <em key={key}>{renderInline(token.value, key)}</em>
      case 'strike':
        return <s key={key}>{renderInline(token.value, key)}</s>
      case 'link':
        // Opens externally via the window-open handler in the main process.
        return (
          <a href={token.href} key={key} target="_blank" rel="noreferrer noopener">
            {token.value}
          </a>
        )
      default:
        return <Fragment key={key}>{token.value}</Fragment>
    }
  })
}

// ---------------------------------------------------------------- blocks

type Block =
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

function parseBlocks(source: string): Block[] {
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

// ---------------------------------------------------------------- render

function renderList(block: Extract<Block, { kind: 'list' }>, key: string): JSX.Element {
  // Flat items with a depth are turned into real nesting, so indentation is structural.
  const render = (items: { text: string; depth: number }[], depth: number, prefix: string): JSX.Element => {
    const Tag = block.ordered && depth === 0 ? 'ol' : 'ul'
    const children: ReactNode[] = []
    for (let i = 0; i < items.length; ) {
      const item = items[i]
      const nested: { text: string; depth: number }[] = []
      let j = i + 1
      while (j < items.length && items[j].depth > depth) {
        nested.push(items[j])
        j++
      }
      children.push(
        <li key={`${prefix}-${i}`}>
          {renderInline(item.text, `${prefix}-${i}`)}
          {nested.length > 0 && render(nested, depth + 1, `${prefix}-${i}-n`)}
        </li>
      )
      i = j
    }
    return <Tag className="md-list">{children}</Tag>
  }
  const top = block.items.filter((it) => it.depth === 0).length ? 0 : Math.min(...block.items.map((it) => it.depth))
  return <Fragment key={key}>{render(block.items, top, key)}</Fragment>
}

export default function Markdown({ source }: { source: string }): JSX.Element {
  const blocks = parseBlocks(source)

  return (
    <div className="markdown">
      {blocks.map((block, i) => {
        const key = `b${i}`
        switch (block.kind) {
          case 'heading': {
            const Tag = `h${Math.min(block.level + 2, 6)}` as 'h3' | 'h4' | 'h5' | 'h6'
            return (
              <Tag className="md-heading" key={key}>
                {renderInline(block.text, key)}
              </Tag>
            )
          }
          case 'code':
            return (
              <pre className="md-pre" key={key}>
                {block.lang && <span className="md-lang">{block.lang}</span>}
                <code>{block.lines.join('\n')}</code>
              </pre>
            )
          case 'list':
            return renderList(block, key)
          case 'quote':
            return (
              <blockquote className="md-quote" key={key}>
                {renderInline(block.lines.join(' '), key)}
              </blockquote>
            )
          case 'table':
            return (
              <div className="md-table-wrap" key={key}>
                <table className="md-table">
                  <thead>
                    <tr>
                      {block.header.map((h, hi) => (
                        <th key={hi}>{renderInline(h, `${key}-h${hi}`)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {block.rows.map((row, ri) => (
                      <tr key={ri}>
                        {row.map((cell, ci) => (
                          <td key={ci}>{renderInline(cell, `${key}-${ri}-${ci}`)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          case 'hr':
            return <hr className="md-hr" key={key} />
          default:
            return (
              <p className="md-p" key={key}>
                {renderInline(block.lines.join('\n'), key)}
              </p>
            )
        }
      })}
    </div>
  )
}
