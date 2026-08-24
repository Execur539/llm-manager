#!/usr/bin/env node
/**
 * A stand-in for the HuggingFace API and file host.
 *
 * Lets the whole discovery-to-download path run offline and deterministically: search, file
 * listing, quant recommendation, resumable downloads with range requests, mmproj auto-fetch,
 * gated repositories and rate limiting. The app reaches it via LLMM_HF_BASE.
 *
 * Served files are real GGUF bytes built by the fixture generator, so a completed download lands
 * in the library and parses like any other model.
 */

import http from 'node:http'
import { parseArgs } from 'node:util'
import { buildGguf } from './fixtures.mjs'

const { values } = parseArgs({ options: { port: { type: 'string', default: '0' } }, strict: false })

const MB = 1024 * 1024

/** Repositories the mock knows about. Sizes are advertised; bodies are generated on demand. */
const REPOS = {
  'mock-org/Test-27B-GGUF': {
    downloads: 6_674_515,
    likes: 1200,
    lastModified: '2026-08-14T00:00:00.000Z',
    files: [
      { path: 'Test-27B-IQ4_XS.gguf', size: 3 * MB },
      { path: 'Test-27B-Q4_K_M.gguf', size: 4 * MB },
      { path: 'Test-27B-Q5_K_M.gguf', size: 5 * MB },
      { path: 'Test-27B-Q8_0.gguf', size: 8 * MB },
      { path: 'mmproj-Test-27B-Q8_0.gguf', size: 1 * MB },
      { path: 'README.md', size: 2048 }
    ]
  },
  'mock-org/Small-3B-GGUF': {
    downloads: 812_004,
    likes: 210,
    lastModified: '2026-07-02T00:00:00.000Z',
    files: [
      { path: 'Small-3B-Q4_K_M.gguf', size: 2 * MB },
      { path: 'Small-3B-Q8_0.gguf', size: 3 * MB }
    ]
  },
  'mock-org/Gated-70B-GGUF': {
    downloads: 44_100,
    likes: 90,
    lastModified: '2026-06-11T00:00:00.000Z',
    gated: true,
    files: [{ path: 'Gated-70B-Q4_K_M.gguf', size: 6 * MB }]
  },
  'mock-org/Sharded-120B-GGUF': {
    downloads: 12_004,
    likes: 33,
    lastModified: '2026-05-01T00:00:00.000Z',
    files: [
      { path: 'Sharded-120B-Q4_K_M-00001-of-00003.gguf', size: 4 * MB },
      { path: 'Sharded-120B-Q4_K_M-00002-of-00003.gguf', size: 4 * MB },
      { path: 'Sharded-120B-Q4_K_M-00003-of-00003.gguf', size: 4 * MB }
    ]
  }
}

/** Behaviour switches a scenario can flip through query parameters. */
const state = { rateLimited: false, slowBytesPerSec: 0, failAfterBytes: 0 }

function json(res, status, body, headers = {}) {
  const text = JSON.stringify(body)
  res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(text), ...headers })
  res.end(text)
}

/** Deterministic GGUF body padded to the advertised size. */
function bodyFor(repo, filePath) {
  const meta = REPOS[repo]?.files.find((f) => f.path === filePath)
  const size = meta?.size ?? MB
  const isProjector = /mmproj/i.test(filePath)
  const gguf = buildGguf(
    isProjector
      ? { architecture: 'clip', name: 'projector', tensorCount: 1, vocabSize: 16, ssm: false }
      : { name: filePath.replace(/\.gguf$/, ''), vocabSize: 2000 }
  )
  if (gguf.length >= size) return gguf
  // Pad with zeros: tensor data is opaque to the parser, which stops at the directory.
  return Buffer.concat([gguf, Buffer.alloc(size - gguf.length)])
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`)

  // Test control surface.
  if (url.pathname === '/__control') {
    if (url.searchParams.has('rateLimited')) state.rateLimited = url.searchParams.get('rateLimited') === '1'
    if (url.searchParams.has('slow')) state.slowBytesPerSec = Number(url.searchParams.get('slow'))
    if (url.searchParams.has('failAfter')) state.failAfterBytes = Number(url.searchParams.get('failAfter'))
    return json(res, 200, state)
  }

  if (state.rateLimited) {
    return json(res, 429, { error: 'Rate limit reached' }, { 'Retry-After': '30' })
  }

  const token = (req.headers.authorization ?? '').replace(/^Bearer\s+/i, '')

  // GET /api/models?search=...
  if (url.pathname === '/api/models') {
    const query = (url.searchParams.get('search') ?? '').toLowerCase()
    const results = Object.entries(REPOS)
      .filter(([id]) => !query || id.toLowerCase().includes(query))
      .map(([id, r]) => ({
        id,
        downloads: r.downloads,
        likes: r.likes,
        lastModified: r.lastModified,
        tags: ['gguf', 'text-generation'],
        gated: Boolean(r.gated)
      }))
    return json(res, 200, results)
  }

  // GET /api/models/<owner>/<name>/tree/main
  const tree = url.pathname.match(/^\/api\/models\/(.+?)\/tree\/main$/)
  if (tree) {
    const repo = decodeURIComponent(tree[1])
    const entry = REPOS[repo]
    if (!entry) return json(res, 404, { error: 'Repo not found' })
    if (entry.gated && !token) return json(res, 403, { error: 'Gated repo' })
    return json(
      res,
      200,
      entry.files.map((f) => ({ path: f.path, type: 'file', size: f.size, lfs: { size: f.size } }))
    )
  }

  // GET /<owner>/<name>/resolve/main/<file>
  const resolve = url.pathname.match(/^\/(.+?)\/resolve\/main\/(.+)$/)
  if (resolve) {
    const repo = decodeURIComponent(resolve[1])
    const filePath = decodeURIComponent(resolve[2])
    const entry = REPOS[repo]
    if (!entry) return json(res, 404, { error: 'Repo not found' })
    if (entry.gated && !token) return json(res, 403, { error: 'Gated repo' })

    const body = bodyFor(repo, filePath)

    // Range support, so resume genuinely exercises the resume path.
    const range = req.headers.range?.match(/bytes=(\d+)-(\d*)/)
    let start = 0
    let end = body.length - 1
    let status = 200
    const headers = { 'Content-Type': 'application/octet-stream', 'Accept-Ranges': 'bytes' }

    if (range) {
      start = Number(range[1])
      if (range[2]) end = Number(range[2])
      if (start >= body.length) {
        res.writeHead(416, { 'Content-Range': `bytes */${body.length}` })
        return res.end()
      }
      status = 206
      headers['Content-Range'] = `bytes ${start}-${end}/${body.length}`
    }

    const slice = body.subarray(start, end + 1)
    headers['Content-Length'] = String(slice.length)
    res.writeHead(status, headers)

    // Optionally trickle or truncate, to exercise progress and failure handling.
    if (state.failAfterBytes > 0 && slice.length > state.failAfterBytes) {
      res.write(slice.subarray(0, state.failAfterBytes))
      return res.destroy()
    }
    if (state.slowBytesPerSec > 0) {
      const chunk = Math.max(1024, Math.floor(state.slowBytesPerSec / 10))
      for (let i = 0; i < slice.length; i += chunk) {
        if (res.destroyed) return
        res.write(slice.subarray(i, i + chunk))
        await new Promise((r) => setTimeout(r, 100))
      }
      return res.end()
    }
    return res.end(slice)
  }

  json(res, 404, { error: `mock-hf: no route ${url.pathname}` })
})

server.listen(Number(values.port), '127.0.0.1', () => {
  process.stdout.write(`${server.address().port}\n`)
})

for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, () => server.close(() => process.exit(0)))
}
