#!/usr/bin/env node
/**
 * A stand-in for llama-server.
 *
 * Speaks the same HTTP surface the app talks to, so every code path above the process boundary
 * stays real: spawn, health polling, SSE parsing, tool-call fragment accumulation, timings,
 * priority queueing, unload. Only the inference is fake — which is the one part that needs a GPU
 * and the one part whose *content* does not matter for finding UI and lifecycle bugs.
 *
 * The app spawns this instead of the real binary when LLMM_MOCK_LLAMA=1.
 *
 * Behaviour is steered by markers in the prompt, so a test can ask for a specific shape of
 * response without any out-of-band channel:
 *
 *   [[mock:long]]        stream ~800 words, for scroll/compaction/perf behaviour
 *   [[mock:slow]]        250ms between tokens, for navigating away mid-stream
 *   [[mock:tool]]        emit a tool call instead of prose
 *   [[mock:tools:N]]     emit N tool calls in sequence
 *   [[mock:error]]       return HTTP 500 mid-stream
 *   [[mock:stall]]       accept the request and never respond
 *   [[mock:empty]]       stream nothing and finish
 *   [[mock:unicode]]     emoji, RTL, CJK and combining marks, for layout defects
 *   [[mock:markdown]]    headings, lists, tables, code fences
 *   [[mock:longline]]    one very long unbroken token run, for overflow defects
 */

import http from 'node:http'
import { parseArgs } from 'node:util'

const { values } = parseArgs({
  options: {
    port: { type: 'string', default: '0' },
    host: { type: 'string', default: '127.0.0.1' },
    model: { type: 'string', default: 'mock-model.gguf' },
    'ctx-size': { type: 'string', default: '4096' }
  },
  strict: false
})

const PORT = Number(values.port)
const HOST = values.host
const MODEL = String(values.model).split(/[\\/]/).pop()

// Startup delay so the app's health-polling loop is genuinely exercised rather than
// short-circuited by an instantly-ready server.
const READY_DELAY_MS = Number(process.env.LLMM_MOCK_READY_MS ?? 1200)
let ready = false
setTimeout(() => {
  ready = true
  process.stderr.write(`mock-llama: model loaded '${MODEL}'\n`)
}, READY_DELAY_MS)

const DEFAULT_REPLY =
  'A state-space model represents a system through a hidden state that evolves over time, ' +
  'with observations produced from that state at each step. Unlike attention, its memory cost ' +
  'stays constant as the sequence grows, which is why hybrid architectures interleave the two.'

const LONG_REPLY = Array.from(
  { length: 60 },
  (_, i) =>
    `Paragraph ${i + 1}. This sentence exists to produce a long, scrollable response so that ` +
    `virtualisation, autoscroll, context accounting and compaction can all be observed under ` +
    `realistic length. It deliberately varies in length so wrapping is not uniform.`
).join('\n\n')

const UNICODE_REPLY =
  'Emoji: 👋🏽 🎉 🧑‍💻 👩‍👩‍👧‍👦 — zero-width joiners and skin tones.\n' +
  'CJK: 日本語のテキスト、中文文本，한국어 텍스트。\n' +
  'RTL: العربية مع نص إنجليزي mixed inline, then back.\n' +
  'Combining: é vs é (composed vs decomposed), a\u0301, o\u0308.\n' +
  'Math: ∑ ∫ √ ≈ ≠ ∞ ⟨ψ|H|ψ⟩\n' +
  'Box drawing: ┌─┬─┐ │ │ └─┴─┘'

const MARKDOWN_REPLY = `# Heading one

Some prose with **bold**, *italic*, \`inline code\` and a [link](https://example.com).

## Lists

- first item
- second item with a much longer body that should wrap across lines in a narrow pane
  - nested item
1. ordered
2. also ordered

## Table

| column | another column | numeric |
|--------|----------------|---------|
| a      | some text      |   1,234 |
| b      | longer text here | 56,789 |

## Code

\`\`\`typescript
export function example(input: string): number {
  // A long line that should scroll horizontally inside its own container rather than the page
  return input.split('').reduce((total, character) => total + character.charCodeAt(0), 0)
}
\`\`\`
`

const LONGLINE_REPLY =
  'Unbroken: ' + 'A'.repeat(400) + ' and a long path: ' +
  'C:\\Users\\someone\\AppData\\Local\\Very\\Deeply\\Nested\\Directory\\Structure\\That\\Keeps\\Going\\file.gguf'

function marker(messages, name) {
  const text = messages.map((m) => (typeof m.content === 'string' ? m.content : JSON.stringify(m.content))).join(' ')
  return text.includes(`[[mock:${name}]]`)
}

function markerValue(messages, name) {
  const text = messages.map((m) => (typeof m.content === 'string' ? m.content : JSON.stringify(m.content))).join(' ')
  const m = text.match(new RegExp(`\\[\\[mock:${name}:(\\d+)\\]\\]`))
  return m ? Number(m[1]) : null
}

function pickReply(messages) {
  if (marker(messages, 'long')) return LONG_REPLY
  if (marker(messages, 'unicode')) return UNICODE_REPLY
  if (marker(messages, 'markdown')) return MARKDOWN_REPLY
  if (marker(messages, 'longline')) return LONGLINE_REPLY
  if (marker(messages, 'empty')) return ''
  return DEFAULT_REPLY
}

/** Split into token-ish chunks so streaming looks like real token emission. */
function tokenise(text) {
  return text.match(/\s*\S+|\s+/g) ?? []
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = ''
    req.on('data', (c) => {
      data += c
    })
    req.on('end', () => resolve(data))
    req.on('error', reject)
  })
}

function json(res, status, body) {
  const text = JSON.stringify(body)
  res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(text) })
  res.end(text)
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`)

  if (url.pathname === '/health') {
    if (!ready) return json(res, 503, { status: 'loading model' })
    return json(res, 200, { status: 'ok' })
  }

  if (url.pathname === '/tokenize' && req.method === 'POST') {
    const body = JSON.parse((await readBody(req)) || '{}')
    const n = Math.ceil(String(body.content ?? '').length / 4)
    return json(res, 200, { tokens: Array.from({ length: n }, (_, i) => i) })
  }

  if (url.pathname === '/v1/embeddings' && req.method === 'POST') {
    const body = JSON.parse((await readBody(req)) || '{}')
    const inputs = Array.isArray(body.input) ? body.input : [body.input]
    // Deterministic pseudo-embeddings: same text always yields the same vector, and similar
    // text yields similar vectors, so retrieval ordering is meaningful in tests.
    const data = inputs.map((text, index) => {
      const vec = new Array(384).fill(0)
      for (let i = 0; i < String(text).length; i++) {
        vec[String(text).charCodeAt(i) % 384] += 1
      }
      const norm = Math.sqrt(vec.reduce((a, b) => a + b * b, 0)) || 1
      return { object: 'embedding', index, embedding: vec.map((v) => v / norm) }
    })
    return json(res, 200, { object: 'list', data })
  }

  if (url.pathname === '/v1/chat/completions' && req.method === 'POST') {
    const body = JSON.parse((await readBody(req)) || '{}')
    const messages = body.messages ?? []

    if (marker(messages, 'stall')) return // never respond, so cancel/abort can be exercised
    if (marker(messages, 'error')) return json(res, 500, { error: { message: 'mock: simulated server error' } })

    const wantsTool = (marker(messages, 'tool') || markerValue(messages, 'tools') !== null) && body.tools?.length
    const toolCount = markerValue(messages, 'tools') ?? 1
    const delay = marker(messages, 'slow') ? 250 : 12
    const reply = pickReply(messages)
    const id = `chatcmpl-mock-${Date.now().toString(36)}`

    if (!body.stream) {
      const message = wantsTool
        ? {
            role: 'assistant',
            content: '',
            tool_calls: buildToolCalls(body.tools, toolCount)
          }
        : { role: 'assistant', content: reply }
      return json(res, 200, {
        id,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: MODEL,
        choices: [{ index: 0, message, finish_reason: wantsTool ? 'tool_calls' : 'stop' }],
        usage: { prompt_tokens: 100, completion_tokens: tokenise(reply).length }
      })
    }

    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive'
    })

    let aborted = false
    req.on('close', () => {
      aborted = true
    })

    const send = (payload) => res.write(`data: ${JSON.stringify(payload)}\n\n`)
    const frame = (delta, finish = null) => ({
      id,
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model: MODEL,
      choices: [{ index: 0, delta, finish_reason: finish }]
    })

    if (wantsTool) {
      const calls = buildToolCalls(body.tools, toolCount)
      // Emit arguments split across frames, the way llama.cpp does, so the client's fragment
      // accumulation is genuinely exercised rather than receiving one tidy object.
      for (const [index, call] of calls.entries()) {
        send(frame({ tool_calls: [{ index, id: call.id, function: { name: call.function.name, arguments: '' } }] }))
        const args = call.function.arguments
        for (let i = 0; i < args.length; i += 7) {
          if (aborted) return
          send(frame({ tool_calls: [{ index, function: { arguments: args.slice(i, i + 7) } }] }))
          await sleep(6)
        }
      }
      send(frame({}, 'tool_calls'))
      res.write('data: [DONE]\n\n')
      return res.end()
    }

    send(frame({ role: 'assistant', content: '' }))
    for (const token of tokenise(reply)) {
      if (aborted) return
      send(frame({ content: token }))
      await sleep(delay)
    }
    send(frame({}, 'stop'))
    res.write('data: [DONE]\n\n')
    return res.end()
  }

  json(res, 404, { error: { message: `mock-llama: no route ${url.pathname}` } })
})

function buildToolCalls(tools, count) {
  const preferred = ['list_dir', 'read_file', 'glob', 'web_search']
  const chosen = []
  for (let i = 0; i < count; i++) {
    const tool =
      tools.find((t) => t.function?.name === preferred[i % preferred.length]) ?? tools[i % tools.length]
    const name = tool.function?.name ?? 'list_dir'
    const args = argsFor(name)
    chosen.push({ id: `call_mock_${i}`, type: 'function', function: { name, arguments: JSON.stringify(args) } })
  }
  return chosen
}

/** Plausible arguments per tool, so approval prompts render realistically. */
function argsFor(name) {
  switch (name) {
    case 'read_file': return { path: 'package.json' }
    case 'glob': return { pattern: 'src/**/*.ts' }
    case 'grep': return { pattern: 'TODO', path: 'src' }
    case 'write_file': return { path: 'mock-output.txt', content: 'written by the mock model' }
    case 'run_command': return { command: 'echo hello from the mock' }
    case 'web_search': return { query: 'state space models' }
    case 'fetch_url': return { url: 'https://example.com' }
    default: return { path: '.' }
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

server.listen(PORT, HOST, () => {
  const addr = server.address()
  process.stderr.write(`mock-llama: listening on ${HOST}:${addr.port}, model '${MODEL}'\n`)
})

for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, () => {
    server.close(() => process.exit(0))
    setTimeout(() => process.exit(0), 500)
  })
}
