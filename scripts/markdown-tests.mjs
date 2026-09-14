/**
 * Markdown grammar tests.
 *
 * The parser is pure and lives apart from the component precisely so it can be checked here:
 * list nesting, ordered-versus-unordered boundaries, unterminated fences and inline precedence
 * are all easy to get subtly wrong, and verifying them through rendered pixels is slow and
 * indirect.
 */

export async function runMarkdownTests(check, section) {
  const { parseBlocks, parseInline, splitSettled, streamingParser } = await import('./built/markdown.js')

  const blocks = (src) => parseBlocks(src)
  const kinds = (src) => blocks(src).map((b) => b.kind).join(',')

  section('Markdown: block structure')

  const headings = blocks('# one\n\n### three')
  check('heading levels are captured', headings[0].level === 1 && headings[1].level === 3)
  check('paragraph is the fallback', kinds('just some prose') === 'p')
  check('horizontal rule', kinds('---') === 'hr')
  check('blockquote', kinds('> quoted') === 'quote')
  check('empty input yields no blocks', blocks('').length === 0)
  check('blank lines do not create blocks', blocks('\n\n\n').length === 0)

  // The defect this guards: a change of marker was absorbed into the previous list, so
  // "1. one" following "- a" rendered as a bullet and lost its number.
  const mixed = blocks('- a\n- b\n1. one\n2. two')
  check('unordered and ordered lists do not merge', mixed.length === 2, kinds('- a\n- b\n1. one\n2. two'))
  check('the first list stays unordered', mixed[0].ordered === false)
  check('the second list is ordered', mixed[1]?.ordered === true)
  check('no items are lost across the split', mixed[0].items.length === 2 && mixed[1].items.length === 2)
  check('ordered-then-unordered also splits', blocks('1. a\n- b').length === 2)

  // Indentation is structural, not decorative.
  const nested = blocks('- a\n  - b\n    - c')[0]
  check('nesting depth is captured', nested.items.map((i) => i.depth).join() === '0,1,2')
  check('a nested marker change does not split', blocks('- a\n  1. b').length === 1)

  // An unterminated fence is exactly what a streaming reply looks like mid-block.
  const openFence = blocks('```ts\nconst x = 1')
  check('unterminated code fence still parses as code', openFence[0].kind === 'code')
  check('code fence keeps its language', openFence[0].lang === 'ts')
  check('closed fence excludes its delimiters', blocks('```\nbody\n```')[0].lines.join() === 'body')
  check('markup inside a fence is not parsed', blocks('```\n# not a heading\n```')[0].lines[0] === '# not a heading')

  const table = blocks('| a | b |\n|---|---|\n| 1 | 2 |')[0]
  check('table header parsed', table.kind === 'table' && table.header.join() === 'a,b')
  check('table rows parsed', table.rows.length === 1 && table.rows[0].join() === '1,2')
  check('a lone pipe line is not a table', blocks('a | b')[0].kind === 'p')

  section('Markdown: inline grammar')

  const types = (s) => parseInline(s).map((t) => t.type).join(',')
  const values = (s) => parseInline(s).map((t) => t.value)

  check('bold', types('**x**') === 'bold')
  check('italic with asterisk', types('*x*') === 'italic')
  check('strikethrough', types('~~x~~') === 'strike')
  check('inline code', types('`x`') === 'code')
  check('plain text is a single token', types('hello world') === 'text')
  check('mixed sequence keeps order', types('a **b** c `d`') === 'text,bold,text,code')

  // Code binds tightest, which matters when the model explains markdown itself.
  check('markup inside code stays literal', values('`**not bold**`')[0] === '**not bold**')

  // Identifiers must survive: snake_case turning into italics is a classic renderer bug.
  check('underscores mid-word are not emphasis', types('a_b_c') === 'text', types('a_b_c'))
  check('a bare asterisk is literal', types('2 * 3') === 'text', types('2 * 3'))

  const link = parseInline('[text](https://example.com)')[0]
  check('link text and href are separated', link.type === 'link' && link.value === 'text' && link.href === 'https://example.com')

  // The renderer only turns http(s) into anchors, but the parser should still tokenise it so
  // the label is shown rather than the raw target.
  const script = parseInline('[click](javascript:alert(1))')[0]
  check('a non-http link is still parsed as a link token', script.type === 'link')
  check('and its label is preserved', script.value === 'click')

  check('unclosed emphasis does not swallow the line', types('**unterminated') === 'text', types('**unterminated'))

  section('Markdown: a streaming reply settles without changing what it renders')

  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b)
  const reply = [
    '# Plan',
    '',
    'First, a paragraph that runs',
    'across two lines.',
    '',
    '- one',
    '- two',
    '  - nested',
    '1. ordered',
    '',
    '> a quote',
    '> that continues',
    '',
    '```ts',
    'const a = 1',
    '',
    'const b = 2',
    '```',
    '',
    '| a | b |',
    '|---|---|',
    '| 1 | 2 |',
    '',
    '---',
    'Closing words with **bold** and `code`.',
    '',
    '  # an indented hash',
    '',
    'after | it',
    '',
    'Last.'
  ].join('\n')

  const split = splitSettled(reply)
  check('the settled part and the tail make up the whole reply', split.settled + split.tail === reply)
  check('a long reply settles most of itself', split.settled.length > reply.length / 2, `${split.settled.length} of ${reply.length}`)
  check('the two parts parse to what the whole does', same([...parseBlocks(split.settled), ...parseBlocks(split.tail)], parseBlocks(reply)))

  const fenced = splitSettled('p1\n\np2\n\np3\n\n```\na\n\nb\n\nc\n\nd')
  check('an open code fence is never cut, whatever blank lines it holds', fenced.tail.includes('```\na\n\nb\n\nc\n\nd') && !fenced.settled.includes('```'))
  check('a reply of one or two blocks settles nothing', splitSettled('just\n\ntwo').settled === '')

  // Every step of a reply arriving a character at a time renders exactly what parsing it whole would.
  const paragraphs = Array.from({ length: 24 }, (_, i) => `Paragraph ${i} with *some* text${i % 5 ? '' : '\n- and\n- a list'}.`).join('\n\n')
  const documents = { reply, crlf: reply.replace(/\n/g, '\r\n'), paragraphs }
  for (const [name, doc] of Object.entries(documents)) {
    for (const keep of [1, 2]) {
      const parse = streamingParser(keep)
      let differsAt = -1
      for (let n = 0; n <= doc.length; n++) {
        const { settled, live } = parse(doc.slice(0, n))
        if (!same([...settled, ...live], parseBlocks(doc.slice(0, n)))) {
          differsAt = n
          break
        }
      }
      check(`streamed a character at a time, ${name} renders as it would whole (keeping ${keep} live)`, differsAt === -1, `first differs at ${differsAt}`)
    }
  }

  const stable = streamingParser()
  const earlier = stable(paragraphs.slice(0, 400)).settled
  const later = stable(paragraphs.slice(0, 600)).settled
  check('blocks that have settled come back as the same objects', earlier.length > 0 && later.length > earlier.length && earlier.every((b, i) => later[i] === b))

  const restarted = streamingParser()
  restarted(reply)
  const other = restarted('Another\n\nreply entirely')
  check('a different reply starts over', same([...other.settled, ...other.live], parseBlocks('Another\n\nreply entirely')))
}
