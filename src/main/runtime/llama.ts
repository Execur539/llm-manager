/**
 * llama-server supervision and the OpenAI-compatible client used to talk to it.
 *
 * The child always binds to 127.0.0.1 on an ephemeral port — it is never exposed directly.
 * Anything reaching it from outside the machine goes through our own authenticated layer.
 */

import { spawn, ChildProcess } from 'node:child_process'
import net from 'node:net'
import { EventEmitter } from 'node:events'
import type { FitPlan, ModelRecord } from '@shared/types'
import { childEnv, llamaServerPath } from './binaries'
import type { Backend } from '@shared/types'

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

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string
  tool_call_id?: string
  name?: string
}

export interface CompletionOptions {
  messages: ChatMessage[]
  temperature?: number
  topP?: number
  maxTokens?: number
  /** GBNF grammar to constrain sampling — how tool calls are made structurally valid */
  grammar?: string
  stop?: string[]
  signal?: AbortSignal
}

export class LlamaRuntime extends EventEmitter {
  private child: ChildProcess | null = null
  private current: LoadedModel | null = null
  private starting: Promise<LoadedModel> | null = null

  get loaded(): LoadedModel | null {
    return this.current
  }

  /** Build the llama-server argv from a fit plan. Every value here came from the engine. */
  private buildArgs(model: ModelRecord, plan: FitPlan, port: number): string[] {
    const args = [
      '--model', model.path,
      '--host', '127.0.0.1',
      '--port', String(port),
      '--ctx-size', String(plan.contextLength),
      '--n-gpu-layers', String(plan.gpuLayers),
      '--batch-size', String(plan.batchSize),
      // Enable Jinja chat templates so native tool-calling handlers are used where the
      // model's template supports them.
      '--jinja'
    ]

    if (plan.kvType !== 'f16') {
      args.push('--cache-type-k', plan.kvType, '--cache-type-v', plan.kvType)
    }
    if (plan.flashAttention) args.push('--flash-attn', 'on')
    if (plan.tensorSplit.length > 1) {
      args.push('--tensor-split', plan.tensorSplit.map((s) => s.toFixed(3)).join(','))
    }
    if (model.caps.mmprojPath) {
      args.push('--mmproj', model.caps.mmprojPath)
    }
    return args
  }

  async load(model: ModelRecord, plan: FitPlan, backend: Backend): Promise<LoadedModel> {
    if (this.starting) await this.starting.catch(() => undefined)
    await this.unload()

    this.starting = (async () => {
      const port = await freePort()
      const exe = llamaServerPath(backend)
      const args = this.buildArgs(model, plan, port)

      this.emit('status', { phase: 'starting', model: model.filename, args })

      const child = spawn(exe, args, { windowsHide: true, env: childEnv() })
      this.child = child

      let stderr = ''
      child.stderr?.on('data', (d: Buffer) => {
        const text = d.toString()
        stderr += text
        if (stderr.length > 256 * 1024) stderr = stderr.slice(-128 * 1024)
        this.emit('log', text)
      })
      child.stdout?.on('data', (d: Buffer) => this.emit('log', d.toString()))
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

  /** Poll /health until the server is up, failing fast if the child dies first. */
  private async waitForHealth(port: number, child: ChildProcess, stderr: () => string): Promise<void> {
    const deadline = Date.now() + 10 * 60 * 1000 // large models genuinely take minutes to load
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
      const done = (): void => resolve()
      child.once('exit', done)
      child.kill()
      // Escalate if it will not go quietly.
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

  /** Streaming chat completion. Yields content deltas as they arrive. */
  async *stream(opts: CompletionOptions): AsyncGenerator<string, void, unknown> {
    const loaded = this.current
    if (!loaded) throw new Error('No model is loaded')

    const body: Record<string, unknown> = {
      messages: opts.messages,
      stream: true,
      temperature: opts.temperature ?? 0.7,
      top_p: opts.topP ?? 0.95,
      n_predict: opts.maxTokens ?? -1
    }
    if (opts.grammar) body.grammar = opts.grammar
    if (opts.stop?.length) body.stop = opts.stop

    const res = await fetch(`http://127.0.0.1:${loaded.port}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: opts.signal
    })
    if (!res.ok || !res.body) {
      throw new Error(`Completion failed: HTTP ${res.status} ${await res.text().catch(() => '')}`)
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // Server-sent events: one JSON object per "data:" line.
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed.startsWith('data:')) continue
        const payload = trimmed.slice(5).trim()
        if (payload === '[DONE]') return
        try {
          const json = JSON.parse(payload) as { choices?: { delta?: { content?: string } }[] }
          const delta = json.choices?.[0]?.delta?.content
          if (delta) yield delta
        } catch {
          /* partial frame; the next chunk completes it */
        }
      }
    }
  }

  /** Non-streaming completion, used for tool-call turns where we need the whole object. */
  async complete(opts: CompletionOptions): Promise<string> {
    let out = ''
    for await (const chunk of this.stream(opts)) out += chunk
    return out
  }
}

export const llama = new LlamaRuntime()
