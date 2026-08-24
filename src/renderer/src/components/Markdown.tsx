import { Fragment, type JSX, type ReactNode } from 'react'
import { parseBlocks, parseInline, type Block } from '../lib/markdown'

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

// Parsing lives in ../lib/markdown so it can be unit-tested without React.

/**
 * Turn inline tokens into elements.
 *
 * Every value arrives as a text node, so React escapes it. There is no path here by which model
 * output becomes markup.
 */
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
        // Only http(s) links become anchors; anything else stays inert text, so a model cannot
        // emit a javascript: or file: target and have it rendered as clickable.
        return /^https?:\/\//i.test(token.href ?? '') ? (
          <a href={token.href} key={key} target="_blank" rel="noreferrer noopener">
            {token.value}
          </a>
        ) : (
          <Fragment key={key}>{token.value}</Fragment>
        )
      default:
        return <Fragment key={key}>{token.value}</Fragment>
    }
  })
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
