/**
 * llama-server supervision and the OpenAI-compatible client used to talk to it.
 *
 * The child always binds to 127.0.0.1 on an ephemeral port — it is never exposed directly.
 * Anything reaching it from outside the machine goes through our own authenticated layer.
 *
 * Tool calling goes through llama.cpp's native OpenAI-style `tools` parameter (enabled by
 * `--jinja`), which uses the model's own template handler where one exists and a generic
 * handler otherwise. A GBNF grammar is attached as well, so a model whose template knows
 * nothing about tools is still constrained to emit a structurally valid call.
 */

import { spawn, ChildProcess } from 'node:child_process'
import net from 'node:net'
import { EventEmitter } from 'node:events'
import type { Backend, FitPlan, ModelRecord, ToolDefinition } from '@shared/types'
import path from 'node:path'
import { app } from 'electron'
import { childEnv, llamaServerPath } from './binaries'

/** Location of the stand-in server used when LLMM_MOCK_LLAMA=1. */
function mockServerPath(): string {
  return process.env.LLMM_MOCK_SCRIPT ?? path.join(app.getAppPath(), 'scripts', 'mock-llama.mjs')
}

async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = net.createServer()
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address()
      if (typeof addr === 'object' && addr) {
        const port = addr.port
        srv.close(() => resolve(port))
      } else {
        srv.close(() => reject(new Error('could not allocate a port')))
      }
    })
    srv.on('error', reject)
  })
}

export interface LoadedModel {
  model: ModelRecord
  plan: FitPlan
  port: number
  startedAt: number
}

export interface ContentPart {
  type: 'text' | 'image_url' | 'input_audio'
  text?: string
  image_url?: { url: string }
  input_audio?: { data: string; format: string }
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string | ContentPart[]
  tool_call_id?: string
  name?: string
  tool_calls?: { id: string; type: 'function'; function: { name: string; arguments: string } }[]
}

export interface CompletionOptions {
  messages: ChatMessage[]
  temperature?: number
  topP?: number
  topK?: number
  minP?: number
  repeatPenalty?: number
  maxTokens?: number
  /** OpenAI-style tool definitions; llama.cpp maps these onto the model's template */
  tools?: ToolDefinition[]
  /** GBNF grammar to constrain sampling — the safety net for weak tool models */
  grammar?: string
  stop?: string[]
  /**
   * Reasoning effort, as named by the model's own chat template.
   *
   * llama-server does not interpret this: it hands the value to the template, which is why the
   * level names differ per model (Qwen3.8 wants 'xhigh', gpt-oss wants 'high'). Passing a name
   * the template does not recognise makes it raise, so this must come from the detected set.
   * The literal 'none' is llama.cpp's own switch for turning thinking off.
   */
  reasoningEffort?: string
  /** Extra Jinja variables — used for `enable_thinking` on models with a plain toggle. */
  chatTemplateKwargs?: Record<string, unknown>
  /**
   * Token budget for thinking; 0 ends it immediately.
   *
   * This one is llama.cpp's, not the template's — it closes the thought block itself rather than
   * asking the template to skip it. That makes it the only way to stop a model thinking when its
   * template offers no switch of its own, which is the common case for effort-only templates
   * like Qwen3.8's: they enumerate levels and provide no way to say "none".
   */
  reasoningBudget?: number
  signal?: AbortSignal
}

export interface StreamedToolCall {
  id: string
  name: string
  args: Record<string, unknown>
}

export type StreamEvent =
  | { type: 'text'; text: string }
  /**
   * The model's chain of thought.
   *
   * llama.cpp defaults to `--reasoning-format deepseek`, which strips <think> blocks out of the
   * answer and returns them separately as `reasoning_content`. Reading only `content` therefore
   * discarded the whole of a reasoning model's thinking without leaving any trace that it had
   * happened — the answers looked clean, and the thought process was simply gone.
   */
  | { type: 'reasoning'; text: string }
  | { type: 'tool_call'; call: StreamedToolCall }
  | { type: 'usage'; promptTokens: number; completionTokens: number }

export interface Timings {
  ttftMs: number | null
  totalMs: number
  /**
   * Tokens in the prompt, as reported by the server.
   *
   * Needed for an OpenAI-shaped `usage` block: clients that track cost read prompt_tokens and
   * total_tokens, and a response carrying only completion_tokens makes them report undefined.
   */
  promptTokens: number
  completionTokens: number
  tokensPerSecond: number
}

export class LlamaRuntime extends EventEmitter {
  private child: ChildProcess | null = null
  private current: LoadedModel | null = null
  private starting: Promise<LoadedModel> | null = null
  private lastTimings: Timings | null = null
  /** Serialises requests so the local user's work is never interleaved with remote work. */
  private queue: Promise<unknown> = Promise.resolve()

  get loaded(): LoadedModel | null {
    return this.current
  }

  get timings(): Timings | null {
    return this.lastTimings
  }

  private buildArgs(model: ModelRecord, plan: FitPlan, port: number): string[] {
    const args = [
      '--model', model.path,
      '--host', '127.0.0.1',
      '--port', String(port),
      '--ctx-size', String(plan.contextLength),
      '--n-gpu-layers', String(plan.gpuLayers),
      '--batch-size', String(plan.batchSize),
      // llama-server defaults to 4 parallel slots. We serialise every request through a
      // priority queue instead, so extra slots buy nothing — and they cost real memory:
      // slots divide the context budget, and on hybrid models the recurrent-state cache is
      // allocated *per slot*, which is enough to turn a fitting plan into an OOM.
      '--parallel', '1',
      // Jinja templates enable llama.cpp's native tool-calling handlers.
      '--jinja'
    ]

    if (plan.kvType !== 'f16') {
      args.push('--cache-type-k', plan.kvType, '--cache-type-v', plan.kvType)
    }
    if (plan.flashAttention) args.push('--flash-attn', 'on')
    if (plan.tensorSplit.length > 1) {
      args.push('--tensor-split', plan.tensorSplit.map((s) => s.toFixed(3)).join(','))
    }
    if (model.caps.mmprojPath) args.push('--mmproj', model.caps.mmprojPath)
    return args
  }

  async load(model: ModelRecord, plan: FitPlan, backend: Backend): Promise<LoadedModel> {
    if (this.starting) await this.starting.catch(() => undefined)
    await this.unload()

    this.starting = (async () => {
      const port = await freePort()
      const args = this.buildArgs(model, plan, port)

      this.emit('status', { phase: 'starting', model: model.filename, args })

      // Test mode: swap in a stand-in server that speaks the same HTTP surface. Everything
      // above the process boundary — health polling, SSE parsing, tool-call accumulation,
      // timings, unload — runs unchanged; only the inference is fake. Guarded by an env var so
      // it can never engage in a normal run.
      const useMock = process.env.LLMM_MOCK_LLAMA === '1'
      const exe = useMock ? process.execPath : llamaServerPath(backend)
      const spawnArgs = useMock ? [mockServerPath(), ...args] : args
      const env = useMock ? { ...childEnv(), ELECTRON_RUN_AS_NODE: '1' } : childEnv()

      const child = spawn(exe, spawnArgs, { windowsHide: true, env })
      this.child = child

      let stderr = ''
      const capture = (d: Buffer): void => {
        const text = d.toString()
        stderr += text
        if (stderr.length > 256 * 1024) stderr = stderr.slice(-128 * 1024)
        this.emit('log', text)
      }
      child.stderr?.on('data', capture)
      child.stdout?.on('data', capture)
      child.on('error', (err) => {
        stderr += `\nspawn error: ${err.message}`
      })
      child.on('exit', (code) => {
        this.emit('status', { phase: 'exited', code })
        if (this.current?.port === port) this.current = null
        this.child = null
      })

      await this.waitForHealth(port, child, () => stderr)

      const loaded: LoadedModel = { model, plan, port, startedAt: Date.now() }
      this.current = loaded
      this.emit('status', { phase: 'ready', model: model.filename, port })
      return loaded
    })()

    try {
      return await this.starting
    } finally {
      this.starting = null
    }
  }

  private async waitForHealth(port: number, child: ChildProcess, stderr: () => string): Promise<void> {
    const deadline = Date.now() + 10 * 60 * 1000 // big models genuinely take minutes
    while (Date.now() < deadline) {
      if (child.exitCode !== null) {
        throw new Error(`llama-server exited with code ${child.exitCode}.\n${stderr().slice(-2000)}`)
      }
      try {
        const res = await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(2000) })
        if (res.ok) return
      } catch {
        /* not up yet */
      }
      await new Promise((r) => setTimeout(r, 500))
    }
    throw new Error('llama-server did not become healthy within 10 minutes')
  }

  async unload(): Promise<void> {
    const child = this.child
    if (!child) {
      this.current = null
      return
    }
    this.child = null
    this.current = null
    await new Promise<void>((resolve) => {
      child.once('exit', () => resolve())
      child.kill()
      setTimeout(() => {
        try {
          child.kill('SIGKILL')
        } catch {
          /* already gone */
        }
        resolve()
      }, 4000)
    })
  }

  /**
   * Enqueue work so concurrent callers (local UI, API server, remote web UI) are serialised
   * rather than interleaving on one llama-server slot.
   */
  enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const next = this.queue.then(fn, fn)
    this.queue = next.catch(() => undefined)
    return next
  }

  private requestBody(opts: CompletionOptions, stream: boolean): Record<string, unknown> {
    const body: Record<string, unknown> = {
      messages: opts.messages,
      stream,
      temperature: opts.temperature ?? 0.7,
      top_p: opts.topP ?? 0.95,
      cache_prompt: true
    }
    if (opts.topK !== undefined) body.top_k = opts.topK
    if (opts.minP !== undefined) body.min_p = opts.minP
    if (opts.repeatPenalty !== undefined) body.repeat_penalty = opts.repeatPenalty
    if (opts.maxTokens !== undefined && opts.maxTokens > 0) body.max_tokens = opts.maxTokens
    if (opts.stop?.length) body.stop = opts.stop
    if (opts.reasoningEffort) body.reasoning_effort = opts.reasoningEffort
    if (opts.chatTemplateKwargs && Object.keys(opts.chatTemplateKwargs).length > 0) {
      body.chat_template_kwargs = opts.chatTemplateKwargs
    }
    // Sent under both names: the server has accepted `reasoning_budget` and `thinking_budget`
    // at different points, and an unrecognised field is ignored rather than rejected.
    if (opts.reasoningBudget !== undefined) {
      body.reasoning_budget = opts.reasoningBudget
      body.thinking_budget = opts.reasoningBudget
    }
    if (opts.grammar) body.grammar = opts.grammar
    if (opts.tools?.length) {
      body.tools = opts.tools.map((t) => ({
        type: 'function',
        function: { name: t.name, description: t.description, parameters: t.parameters }
      }))
      body.tool_choice = 'auto'
    }
    return body
  }

  /**
   * Streaming completion yielding typed events.
   *
   * llama.cpp streams tool calls as incremental `delta.tool_calls` fragments, with the
   * arguments arriving as a partial JSON string across many frames — so fragments are
   * accumulated per index and only parsed once the stream ends.
   */
  async *streamEvents(opts: CompletionOptions): AsyncGenerator<StreamEvent, void, unknown> {
    const loaded = this.current
    if (!loaded) throw new Error('No model is loaded')

    const startedAt = Date.now()
    let ttft: number | null = null
    let completionTokens = 0
    let promptTokens = 0

    const res = await fetch(`http://127.0.0.1:${loaded.port}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(this.requestBody(opts, true)),
      signal: opts.signal
    })
    if (!res.ok || !res.body) {
      throw new Error(`Completion failed: HTTP ${res.status} ${await res.text().catch(() => '')}`)
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    // index -> accumulating call
    const pending = new Map<number, { id: string; name: string; args: string }>()

    const flushCalls = function* (): Generator<StreamEvent> {
      for (const [, c] of pending) {
        let parsed: Record<string, unknown> = {}
        try {
          parsed = c.args.trim() ? (JSON.parse(c.args) as Record<string, unknown>) : {}
        } catch {
          // Grammar-constrained output should not land here, but a template-native handler
          // can still emit something unparsable; surface it as an empty-arg call so the
          // loop reports a usable error rather than throwing.
          parsed = {}
        }
        yield { type: 'tool_call', call: { id: c.id, name: c.name, args: parsed } }
      }
      pending.clear()
    }

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed.startsWith('data:')) continue
          const payload = trimmed.slice(5).trim()
          if (payload === '[DONE]') {
            yield* flushCalls()
            this.lastTimings = finalise(startedAt, ttft, completionTokens, promptTokens)
            return
          }

          let json: {
            choices?: {
              delta?: {
                content?: string
                reasoning_content?: string
                tool_calls?: { index?: number; id?: string; function?: { name?: string; arguments?: string } }[]
              }
              finish_reason?: string
            }[]
            usage?: { prompt_tokens?: number; completion_tokens?: number }
          }
          try {
            json = JSON.parse(payload)
          } catch {
            continue // partial frame; the next chunk completes it
          }

          const choice = json.choices?.[0]
          const delta = choice?.delta

          if (delta?.reasoning_content) {
            // Thinking counts towards time-to-first-token: it is the model working, and a
            // reasoning model can spend a long time here before any answer appears.
            if (ttft === null) ttft = Date.now() - startedAt
            yield { type: 'reasoning', text: delta.reasoning_content }
          }

          if (delta?.content) {
            if (ttft === null) ttft = Date.now() - startedAt
            completionTokens += estimateTokens(delta.content)
            yield { type: 'text', text: delta.content }
          }

          if (delta?.tool_calls) {
            for (const frag of delta.tool_calls) {
              const idx = frag.index ?? 0
              const existing = pending.get(idx) ?? { id: frag.id ?? `call_${idx}`, name: '', args: '' }
              if (frag.id) existing.id = frag.id
              if (frag.function?.name) existing.name += frag.function.name
              if (frag.function?.arguments) existing.args += frag.function.arguments
              pending.set(idx, existing)
            }
          }

          if (json.usage) {
            promptTokens = json.usage.prompt_tokens ?? promptTokens
            yield {
              type: 'usage',
              promptTokens,
              completionTokens: json.usage.completion_tokens ?? completionTokens
            }
          }

          if (choice?.finish_reason) {
            yield* flushCalls()
          }
        }
      }
      yield* flushCalls()
      this.lastTimings = finalise(startedAt, ttft, completionTokens)
    } finally {
      try {
        await reader.cancel()
      } catch {
        /* already closed */
      }
    }
  }

  /** Text-only stream, for plain chat where tool calls are not wanted. */
  async *stream(opts: CompletionOptions): AsyncGenerator<string, void, unknown> {
    for await (const ev of this.streamEvents(opts)) {
      if (ev.type === 'text') yield ev.text
    }
  }

  async complete(opts: CompletionOptions): Promise<string> {
    let out = ''
    for await (const chunk of this.stream(opts)) out += chunk
    return out
  }

  /** Embeddings via the same server; used by RAG when the embedding model is loaded. */
  async embed(texts: string[], port?: number): Promise<number[][]> {
    const target = port ?? this.current?.port
    if (!target) throw new Error('No model is loaded for embeddings')
    const res = await fetch(`http://127.0.0.1:${target}/v1/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input: texts })
    })
    if (!res.ok) throw new Error(`Embeddings failed: HTTP ${res.status}`)
    const json = (await res.json()) as { data: { embedding: number[] }[] }
    return json.data.map((d) => d.embedding)
  }

  /** Ask the server how many tokens a string costs — used for compaction decisions. */
  async tokenCount(text: string): Promise<number> {
    const loaded = this.current
    if (!loaded) return estimateTokens(text)
    try {
      const res = await fetch(`http://127.0.0.1:${loaded.port}/tokenize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: text })
      })
      if (!res.ok) return estimateTokens(text)
      const json = (await res.json()) as { tokens: number[] }
      return json.tokens.length
    } catch {
      return estimateTokens(text)
    }
  }
}

function finalise(startedAt: number, ttft: number | null, completionTokens: number, promptTokens = 0): Timings {
  const totalMs = Date.now() - startedAt
  const genMs = ttft === null ? totalMs : Math.max(1, totalMs - ttft)
  return {
    ttftMs: ttft,
    totalMs,
    promptTokens,
    completionTokens,
    tokensPerSecond: completionTokens > 0 ? (completionTokens / genMs) * 1000 : 0
  }
}

/** Rough token estimate for when the server cannot be asked. */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4)
}

export const llama = new LlamaRuntime()
