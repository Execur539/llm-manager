/**
 * The OpenAI-compatible API server, plus an Anthropic Messages endpoint.
 *
 * Two things make this more than a proxy to llama-server:
 *   - JIT model loading: a request naming a model that is not resident loads it first, so
 *     external tools "just work" without the user switching models by hand.
 *   - Local-user priority: desktop activity is never queued behind a remote request. The
 *     runtime serialises work, and local requests jump ahead of remote ones.
 *
 * Anthropic support exists because llama.cpp now speaks that API too, which means a
 * Claude-compatible client can point at LLM Manager as its backend.
 */

import http from 'node:http'
import crypto from 'node:crypto'
import type { AddressInfo } from 'node:net'
import { llama, type ChatMessage } from '../runtime/llama'
import { run } from '../storage/db'

export interface ApiServerOptions {
  port: number
  apiKey: string | null
  jitLoad: boolean
  /** binds beyond loopback only when the remote feature is on */
  exposeToNetwork: boolean
  /** resolve a model name to something loadable, and load it */
  loadModel: (name: string) => Promise<void>
  listModels: () => { id: string; filename: string; bytes: number }[]
}

export type ClientKind = 'local' | 'remote'

interface QueuedRequest<T> {
  priority: number
  run: () => Promise<T>
}

/**
 * Priority queue: local (0) before remote (1). Within a priority, first come first served.
 * One in flight at a time, matching the one-model-at-a-time decision.
 */
class PriorityQueue {
  private items: (QueuedRequest<unknown> & { resolve: (v: unknown) => void; reject: (e: unknown) => void })[] = []
  private running = false

  enqueue<T>(priority: number, fn: () => Promise<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.items.push({
        priority,
        run: fn as () => Promise<unknown>,
        resolve: resolve as (v: unknown) => void,
        reject
      })
      // Stable insert by priority so local work overtakes queued remote work.
      this.items.sort((a, b) => a.priority - b.priority)
      void this.drain()
    })
  }

  get depth(): number {
    return this.items.length + (this.running ? 1 : 0)
  }

  private async drain(): Promise<void> {
    if (this.running) return
    const next = this.items.shift()
    if (!next) return
    this.running = true
    try {
      next.resolve(await next.run())
    } catch (err) {
      next.reject(err)
    } finally {
      this.running = false
      void this.drain()
    }
  }
}

export const requestQueue = new PriorityQueue()

function logRequest(
  endpoint: string,
  modelId: string | null,
  tokensIn: number,
  tokensOut: number,
  ms: number,
  client: ClientKind,
  ip: string,
  status: number
): void {
  try {
    run(
      'INSERT INTO requests (ts, endpoint, model_id, tokens_in, tokens_out, ms, client, ip, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
      Date.now(),
      endpoint,
      modelId,
      tokensIn,
      tokensOut,
      ms,
      client,
      ip,
      status
    )
  } catch {
    /* logging must never break a request */
  }
}

function json(res: http.ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body)
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(text),
    'Access-Control-Allow-Origin': '*'
  })
  res.end(text)
}

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = ''
    req.on('data', (c: Buffer) => {
      data += c
      if (data.length > 64 * 1024 * 1024) reject(new Error('Request body too large'))
    })
    req.on('end', () => resolve(data))
    req.on('error', reject)
  })
}

/** Anthropic content blocks -> flat text, so both API shapes share one path. */
function flattenAnthropicContent(content: unknown): string {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  return content
    .map((b) => {
      const block = b as { type?: string; text?: string; content?: unknown }
      if (block.type === 'text') return block.text ?? ''
      if (block.type === 'tool_result') return typeof block.content === 'string' ? block.content : JSON.stringify(block.content)
      return ''
    })
    .join('\n')
}

export class ApiServer {
  private server: http.Server | null = null
  private opts: ApiServerOptions | null = null

  get port(): number | null {
    const addr = this.server?.address()
    return addr && typeof addr === 'object' ? (addr as AddressInfo).port : null
  }

  get running(): boolean {
    return !!this.server?.listening
  }

  async start(opts: ApiServerOptions): Promise<number> {
    await this.stop()
    this.opts = opts

    const server = http.createServer((req, res) => {
      void this.handle(req, res).catch((err) => {
        json(res, 500, { error: { message: err instanceof Error ? err.message : String(err) } })
      })
    })

    await new Promise<void>((resolve, reject) => {
      server.once('error', reject)
      server.listen(opts.port, opts.exposeToNetwork ? '0.0.0.0' : '127.0.0.1', () => resolve())
    })

    this.server = server
    return this.port ?? opts.port
  }

  async stop(): Promise<void> {
    const server = this.server
    if (!server) return
    this.server = null
    await new Promise<void>((resolve) => server.close(() => resolve()))
  }

  private authorised(req: http.IncomingMessage): boolean {
    const key = this.opts?.apiKey
    if (!key) return true
    const header = req.headers.authorization ?? ''
    const xApiKey = (req.headers['x-api-key'] as string) ?? ''
    const bearer = header.startsWith('Bearer ') ? header.slice(7) : ''
    const provided = bearer || xApiKey
    if (!provided) return false
    // Constant-time compare so the key cannot be probed by timing.
    const a = Buffer.from(provided)
    const b = Buffer.from(key)
    return a.length === b.length && crypto.timingSafeEqual(a, b)
  }

  private clientKind(req: http.IncomingMessage): ClientKind {
    const ip = req.socket.remoteAddress ?? ''
    return ip === '127.0.0.1' || ip === '::1' || ip === '::ffff:127.0.0.1' ? 'local' : 'remote'
  }

  private async handle(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`)
    const ip = req.socket.remoteAddress ?? 'unknown'
    const client = this.clientKind(req)

    if (req.method === 'OPTIONS') {
      res.writeHead(204, {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-api-key, anthropic-version',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
      })
      res.end()
      return
    }

    if (url.pathname === '/health') {
      json(res, 200, { status: 'ok', loaded: llama.loaded?.model.filename ?? null, queue: requestQueue.depth })
      return
    }

    if (!this.authorised(req)) {
      logRequest(url.pathname, null, 0, 0, 0, client, ip, 401)
      json(res, 401, { error: { message: 'Invalid or missing API key', type: 'authentication_error' } })
      return
    }

    if (url.pathname === '/v1/models' && req.method === 'GET') {
      const models = this.opts?.listModels() ?? []
      json(res, 200, {
        object: 'list',
        data: models.map((m) => ({
          id: m.filename,
          object: 'model',
          created: Math.floor(Date.now() / 1000),
          owned_by: 'llm-manager',
          bytes: m.bytes
        }))
      })
      return
    }

    if (url.pathname === '/v1/chat/completions' && req.method === 'POST') {
      await this.chatCompletions(req, res, client, ip)
      return
    }

    if (url.pathname === '/v1/messages' && req.method === 'POST') {
      await this.anthropicMessages(req, res, client, ip)
      return
    }

    if (url.pathname === '/v1/embeddings' && req.method === 'POST') {
      const started = Date.now()
      const body = JSON.parse(await readBody(req)) as { input: string | string[] }
      const texts = Array.isArray(body.input) ? body.input : [body.input]
      const vectors = await requestQueue.enqueue(client === 'local' ? 0 : 1, () => llama.embed(texts))
      logRequest(url.pathname, null, texts.join('').length / 4, 0, Date.now() - started, client, ip, 200)
      json(res, 200, {
        object: 'list',
        data: vectors.map((embedding, index) => ({ object: 'embedding', index, embedding }))
      })
      return
    }

    json(res, 404, { error: { message: `Unknown endpoint ${url.pathname}` } })
  }

  /** Load the requested model if it is not the resident one. */
  private async ensureModel(requested: string | undefined): Promise<void> {
    if (!requested || !this.opts?.jitLoad) return
    const current = llama.loaded?.model.filename
    if (current === requested) return
    const known = this.opts.listModels().find((m) => m.filename === requested || m.id === requested)
    if (!known) return // unknown name: fall through to whatever is loaded
    await this.opts.loadModel(known.id)
  }

  private async chatCompletions(
    req: http.IncomingMessage,
    res: http.ServerResponse,
    client: ClientKind,
    ip: string
  ): Promise<void> {
    const started = Date.now()
    const body = JSON.parse(await readBody(req)) as {
      model?: string
      messages: ChatMessage[]
      stream?: boolean
      temperature?: number
      top_p?: number
      max_tokens?: number
      tools?: { function: { name: string; description: string; parameters: Record<string, unknown> } }[]
      stop?: string[]
    }

    await this.ensureModel(body.model)
    if (!llama.loaded) {
      json(res, 503, { error: { message: 'No model is loaded and JIT loading did not resolve one.' } })
      return
    }

    const priority = client === 'local' ? 0 : 1
    const tools = body.tools?.map((t) => ({
      name: t.function.name,
      description: t.function.description,
      tier: 'read' as const,
      parameters: t.function.parameters
    }))

    const opts = {
      messages: body.messages,
      temperature: body.temperature,
      topP: body.top_p,
      maxTokens: body.max_tokens,
      stop: body.stop,
      tools
    }

    if (!body.stream) {
      const text = await requestQueue.enqueue(priority, () => llama.complete(opts))
      const t = llama.timings
      logRequest('/v1/chat/completions', llama.loaded.model.filename, 0, t?.completionTokens ?? 0, Date.now() - started, client, ip, 200)
      json(res, 200, {
        id: `chatcmpl-${crypto.randomBytes(8).toString('hex')}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: llama.loaded.model.filename,
        choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'stop' }],
        usage: { completion_tokens: t?.completionTokens ?? 0 }
      })
      return
    }

    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'Access-Control-Allow-Origin': '*'
    })

    const id = `chatcmpl-${crypto.randomBytes(8).toString('hex')}`
    await requestQueue.enqueue(priority, async () => {
      for await (const ev of llama.streamEvents(opts)) {
        if (ev.type !== 'text') continue
        res.write(
          `data: ${JSON.stringify({
            id,
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            model: llama.loaded?.model.filename,
            choices: [{ index: 0, delta: { content: ev.text }, finish_reason: null }]
          })}\n\n`
        )
      }
      res.write(`data: ${JSON.stringify({ id, object: 'chat.completion.chunk', choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\n\n`)
      res.write('data: [DONE]\n\n')
      res.end()
    })

    logRequest('/v1/chat/completions', llama.loaded.model.filename, 0, llama.timings?.completionTokens ?? 0, Date.now() - started, client, ip, 200)
  }

  /** Anthropic Messages API, so Claude-compatible clients can use a local model. */
  private async anthropicMessages(
    req: http.IncomingMessage,
    res: http.ServerResponse,
    client: ClientKind,
    ip: string
  ): Promise<void> {
    const started = Date.now()
    const body = JSON.parse(await readBody(req)) as {
      model?: string
      system?: string | { type: string; text: string }[]
      messages: { role: 'user' | 'assistant'; content: unknown }[]
      stream?: boolean
      max_tokens?: number
      temperature?: number
    }

    await this.ensureModel(body.model)
    if (!llama.loaded) {
      json(res, 503, { error: { type: 'overloaded_error', message: 'No model is loaded.' } })
      return
    }

    const messages: ChatMessage[] = []
    if (body.system) {
      messages.push({
        role: 'system',
        content: typeof body.system === 'string' ? body.system : body.system.map((s) => s.text).join('\n')
      })
    }
    for (const m of body.messages) {
      messages.push({ role: m.role, content: flattenAnthropicContent(m.content) })
    }

    const priority = client === 'local' ? 0 : 1
    const opts = { messages, temperature: body.temperature, maxTokens: body.max_tokens }
    const id = `msg_${crypto.randomBytes(10).toString('hex')}`

    if (!body.stream) {
      const text = await requestQueue.enqueue(priority, () => llama.complete(opts))
      const t = llama.timings
      logRequest('/v1/messages', llama.loaded.model.filename, 0, t?.completionTokens ?? 0, Date.now() - started, client, ip, 200)
      json(res, 200, {
        id,
        type: 'message',
        role: 'assistant',
        model: llama.loaded.model.filename,
        content: [{ type: 'text', text }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 0, output_tokens: t?.completionTokens ?? 0 }
      })
      return
    }

    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'Access-Control-Allow-Origin': '*'
    })

    const sse = (event: string, data: unknown): void => {
      res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
    }

    await requestQueue.enqueue(priority, async () => {
      sse('message_start', {
        type: 'message_start',
        message: { id, type: 'message', role: 'assistant', model: llama.loaded?.model.filename, content: [], usage: { input_tokens: 0, output_tokens: 0 } }
      })
      sse('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } })

      for await (const ev of llama.streamEvents(opts)) {
        if (ev.type !== 'text') continue
        sse('content_block_delta', {
          type: 'content_block_delta',
          index: 0,
          delta: { type: 'text_delta', text: ev.text }
        })
      }

      sse('content_block_stop', { type: 'content_block_stop', index: 0 })
      sse('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: llama.timings?.completionTokens ?? 0 } })
      sse('message_stop', { type: 'message_stop' })
      res.end()
    })

    logRequest('/v1/messages', llama.loaded.model.filename, 0, llama.timings?.completionTokens ?? 0, Date.now() - started, client, ip, 200)
  }
}

export const apiServer = new ApiServer()
