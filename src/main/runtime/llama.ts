/**
 * llama-server supervision and the OpenAI-compatible client used to talk to it.
 *
 * The child always binds to 127.0.0.1 on an ephemeral port — it is never exposed directly.
 * Anything reaching it from outside the machine goes through our own authenticated layer.
 *
 * Tool calling goes through llama.cpp's native OpenAI-style `tools` parameter (enabled by
 * `--jinja`), which uses the model's own template handler where one exists and a generic,
 * grammar-constrained handler otherwise — so a model whose template knows nothing about tools
 * is still held to a structurally valid call. That constraint is llama.cpp's own; this said we
 * attached a GBNF grammar as well, which was never true. `CompletionOptions.grammar` exists and
 * is forwarded, but nothing in the app sets it (see agent/gbnf.ts, which is written and tested
 * but not wired to anything).
 */

import { spawn, ChildProcess } from 'node:child_process'
import net from 'node:net'
import os from 'node:os'
import { EventEmitter } from 'node:events'
import type { Backend, FitPlan, ModelRecord, ToolDefinition } from '@shared/types'
import path from 'node:path'
import { app } from 'electron'
import { childEnv, llamaServerPath } from './binaries'
import { logger } from '../log'

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
  /**
   * Set when the vision projector could not get the VRAM it asked for at load time.
   *
   * llama.cpp treats that allocation failure as non-fatal: it logs the error, carries on, and
   * reports the model loaded. The model then works perfectly for text and segfaults the first
   * time an image or a video reaches it, taking the conversation and the loaded weights with it.
   * Recorded here so media can be refused with an explanation instead.
   */
  visionUnavailable?: string
}

export interface ContentPart {
  type: 'text' | 'image_url' | 'input_audio' | 'input_video'
  text?: string
  image_url?: { url: string }
  input_audio?: { data: string; format: string }
  /**
   * A whole video, handed to the server rather than pre-split into images.
   *
   * llama.cpp expands it through ffmpeg and encodes it as video — pairing consecutive frames
   * into 6-channel super-frames with a 3D convolution, applying temporal M-RoPE, and
   * interleaving timestamps. Sending the same frames as separate `image_url` parts gets none of
   * that: twice the tokens, no motion, and no sense of when anything happened.
   */
  input_video?: { data?: string; url?: string }
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
  /**
   * A tool call the model is still writing.
   *
   * Calls are only complete once their arguments have finished arriving, so a finished
   * `tool_call` cannot be emitted until the stream ends — which means everything between the
   * model deciding to act and the call being ready was silence. On a long argument, a file being
   * written or a command being composed, that is several seconds of the agent appearing to have
   * stopped.
   *
   * `args` is the raw JSON accumulated so far and is therefore usually incomplete: `{"path":"no`
   * is a normal value for it. It is for showing, not for parsing — the finished `tool_call`
   * carries the parsed arguments.
   */
  | { type: 'tool_call_partial'; index: number; name: string; args: string }
  | { type: 'usage'; promptTokens: number; completionTokens: number }
  /**
   * How far the server has got through *reading* the prompt, before generation begins.
   *
   * A long conversation, a big pasted document or a model that has just been loaded can all
   * spend many seconds here producing nothing, which from the outside is indistinguishable
   * from the app having hung. llama.cpp only reports it when asked (`return_progress`).
   *
   * `total` counts the whole prompt and `cached` the prefix served from the KV cache, so a
   * follow-up in a long conversation legitimately starts at a high percentage rather than at
   * zero — that prefix really is already done.
   */
  | { type: 'prompt_progress'; processed: number; total: number; cached: number; percent: number }
  /**
   * How much of the context window this conversation is currently occupying.
   *
   * Both numbers come from the server rather than being estimated here. The prompt total arrives
   * with the first progress frame, well before any output, and generated tokens are added to it
   * as they stream — so the figure is live during a response rather than a number that jumps
   * once the turn is over.
   *
   * Reasoning counts. It occupies the window exactly as answer text does, and on a thinking
   * model it is frequently the larger half.
   */
  | { type: 'context'; used: number; max: number }

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


/**
 * How much system RAM to let llama.cpp use for its prompt cache and context checkpoints.
 *
 * Costs no VRAM and buys back the thing that actually hurts on a long conversation: a hybrid
 * model cannot truncate its cache to resume part-way, so it resumes from a checkpoint or it
 * reprocesses the entire prompt. Each checkpoint is on the order of 150 MB, and the default
 * budget of 8 GB is a handful of them.
 *
 * Measured against free memory rather than total, and capped, for three reasons: the figure is
 * a ceiling llama.cpp fills opportunistically rather than an allocation, other things on the
 * machine need room, and past a few dozen gigabytes the cache is holding conversations nobody
 * will return to. A machine with little free memory gets the default left alone.
 */
function hostCacheBudgetMb(): number {
  const freeMb = Math.floor(os.freemem() / (1024 * 1024))
  const DEFAULT_MB = 8192
  const CAP_MB = 49152
  // Below this there is nothing to give, and taking a quarter of it would hurt.
  if (freeMb < 12288) return 0
  const quarter = Math.floor(freeMb / 4)
  return Math.max(DEFAULT_MB, Math.min(quarter, CAP_MB))
}


export class LlamaRuntime extends EventEmitter {
  private child: ChildProcess | null = null
  private current: LoadedModel | null = null
  private starting: Promise<LoadedModel> | null = null
  private lastTimings: Timings | null = null
  /** Cleared whenever the loaded model changes; see modalities(). */
  private modalityCache: { vision: boolean; audio: boolean; video: boolean } | null = null

  get loaded(): LoadedModel | null {
    return this.current
  }

  get timings(): Timings | null {
    return this.lastTimings
  }

  /**
   * What the running server will actually accept, as it reports it.
   *
   * Distinct from the model's own capabilities. `caps.nativeVideo` says the weights were trained
   * on video; this says the build in front of us can decode one — which additionally needs mtmd
   * video support compiled in, a projector loaded, and ffmpeg findable on PATH. Either can be
   * true without the other, and sending a video to a server that cannot take one is an error the
   * user sees rather than a graceful fallback.
   *
   * Cached per load, since it cannot change while one model is up.
   */
  async modalities(): Promise<{ vision: boolean; audio: boolean; video: boolean }> {
    const none = { vision: false, audio: false, video: false }
    const loaded = this.current
    if (!loaded) return none
    if (this.modalityCache) return this.modalityCache
    try {
      const res = await fetch(`http://127.0.0.1:${loaded.port}/props`, {
        signal: AbortSignal.timeout(5000)
      })
      const json = (await res.json()) as { modalities?: Record<string, boolean> }
      this.modalityCache = {
        vision: !!json.modalities?.vision,
        audio: !!json.modalities?.audio,
        video: !!json.modalities?.video
      }
    } catch {
      // A server that will not answer /props is not one to send a video to either.
      this.modalityCache = none
    }
    return this.modalityCache
  }

  private buildArgs(model: ModelRecord, plan: FitPlan, port: number): string[] {
    const args = [
      '--model', model.path,
      '--host', '127.0.0.1',
      '--port', String(port),
      '--ctx-size', String(plan.contextLength),
      '--n-gpu-layers', String(plan.gpuLayers),
      '--batch-size', String(plan.batchSize),
      // llama-server defaults to 4 parallel slots. One is deliberate: slots divide the context
      // budget between them, and on hybrid models the recurrent-state cache is allocated *per
      // slot*, which is enough to turn a fitting plan into an OOM. Concurrent callers wait —
      // API requests in this process's own priority queue, everything else in the server's.
      '--parallel', '1',
      // Jinja templates enable llama.cpp's native tool-calling handlers.
      '--jinja'
    ]

    /*
     * Keys and values are set separately, because they are not equally sensitive.
     *
     * Both used to take the same type, so buying context meant pushing the keys down as far as
     * the values -- and four-bit keys are the one setting measured to break a model rather than
     * blunt it. The planner now picks them independently and this passes what it picked.
     */
    if (plan.kvType !== 'f16') args.push('--cache-type-k', plan.kvType)
    if ((plan.kvTypeV ?? plan.kvType) !== 'f16') args.push('--cache-type-v', plan.kvTypeV ?? plan.kvType)

    /*
     * Host memory, put to work.
     *
     * This model is hybrid: forty-eight of its sixty-five layers keep a recurrent state rather
     * than a KV cache, and a recurrent state cannot be rolled back to an earlier token the way a
     * KV cache can be truncated. llama.cpp therefore cannot resume mid-conversation from the
     * cache alone -- it resumes from a context checkpoint, and when none covers the resume point
     * it reprocesses the whole prompt from token zero. That is the long silence before a reply on
     * a conversation that has been going a while.
     *
     * Checkpoints live in system RAM, and the defaults are sized for a machine that does not have
     * much: 8 GB of prompt cache, 32 checkpoints, and cache-reuse switched off entirely. On a
     * machine with real headroom those are the wrong numbers, and none of them cost VRAM.
     */
    const cacheRamMb = hostCacheBudgetMb()
    if (cacheRamMb > 0) args.push('--cache-ram', String(cacheRamMb))
    /*
     * Two different mechanisms, and which one applies depends on the architecture.
     *
     * `--cache-reuse` works by shifting the KV cache to line a new prefix up with a cached one.
     * That is only meaningful where the cache is sliceable, so it does nothing for a model with
     * recurrent layers -- a DeltaNet state has to be restored whole at a position, not cut at a
     * token -- and llama.cpp reports as much rather than doing it. Sending it anyway would be
     * noise in the log and a claim in this file that is not true.
     *
     * Hybrid models get context checkpoints instead, which is their only way back into the
     * middle of a conversation without reprocessing from zero. Thirty-two across a long session
     * is thin, and they cost host memory rather than VRAM.
     *
     * `--checkpoint-min-step` is deliberately left alone: ggml-org/llama.cpp#24055 reports
     * checkpoints being invalidated outright on hybrid models when it is set, and it is open.
     */
    const recurrent = (model.arch?.ssmLayers ?? 0) > 0
    if (!recurrent) {
      // 256 tokens is the chunk size the llama.cpp tutorials settle on: large enough not to churn
      // on fragments, small enough to catch the shared block an agent turn actually has.
      args.push('--cache-reuse', '256')
    } else if (cacheRamMb > 0) {
      args.push('--ctx-checkpoints', '64')
    }

    /*
     * Context past what the model was trained for, only when asked for explicitly.
     *
     * YaRN interpolates the rotary positions so the model can address a longer window than it
     * ever saw. It works, and it is not free: llama.cpp applies the scaling statically, at every
     * length, so a short prompt is also being fed positions the model was not trained on. The
     * long-context benchmarks are unkind here even before extension -- most models fall below
     * their own advertised effective length -- so this stays off unless someone turns it on.
     */
    const trained = model.arch?.contextLength ?? 0
    if (trained > 0 && plan.contextLength > trained) {
      const scale = plan.contextLength / trained
      args.push(
        '--rope-scaling', 'yarn',
        '--rope-scale', scale.toFixed(5),
        '--yarn-orig-ctx', String(trained)
      )
      logger.warn(
        'model',
        `context ${plan.contextLength.toLocaleString()} exceeds the trained ${trained.toLocaleString()}; ` +
          `YaRN scaling by ${scale.toFixed(2)}x. Quality past the trained length is not guaranteed.`
      )
    }
    if (plan.flashAttention) args.push('--flash-attn', 'on')
    if (plan.tensorSplit.length > 1) {
      args.push('--tensor-split', plan.tensorSplit.map((s) => s.toFixed(3)).join(','))
    }
    if (model.caps.mmprojPath) args.push('--mmproj', model.caps.mmprojPath)
    return args
  }

  /**
   * Load a model, replacing whatever was resident.
   *
   * Serialised by chaining onto the previous load rather than by checking a flag: the guard used
   * to be `if (this.starting) await it`, followed by an `await this.unload()` before `starting`
   * was assigned. Two callers arriving together — the desktop and a JIT load from the API server
   * — both saw a null `starting`, both awaited the unload, and both spawned. The second child
   * overwrote `this.child`, so the first was never killed: an orphaned llama-server holding a
   * model's worth of VRAM for the life of the machine.
   */
  async load(model: ModelRecord, plan: FitPlan, backend: Backend): Promise<LoadedModel> {
    const previous = this.starting
    const attempt = (async () => {
      await previous?.catch(() => undefined)
      return this.doLoad(model, plan, backend)
    })()
    this.starting = attempt

    try {
      return await attempt
    } finally {
      if (this.starting === attempt) this.starting = null
    }
  }

  private async doLoad(model: ModelRecord, plan: FitPlan, backend: Backend): Promise<LoadedModel> {
    await this.unload()

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
    let spawnFailure: Error | null = null
    const capture = (d: Buffer): void => {
      const text = d.toString()
      stderr += text
      if (stderr.length > 256 * 1024) stderr = stderr.slice(-128 * 1024)
      this.emit('log', text)
    }
    child.stderr?.on('data', capture)
    child.stdout?.on('data', capture)
    child.on('error', (err) => {
      // A process that could not be started never exits, so `exitCode` stays null and the health
      // poll had nothing to notice. A missing or unrunnable llama-server binary therefore looked
      // exactly like a slow load, for ten minutes, before failing with a timeout that said
      // nothing about the real cause.
      spawnFailure = err
      stderr += `\nspawn error: ${err.message}`
    })
    child.on('exit', (code, signal) => {
      /*
       * Cleared before the event, not after.
       *
       * The listener on this is synchronous and asks `llama.loaded` whether a model is still up
       * before telling the interface it is gone. Emitting first meant it was still set, the
       * condition never held, and the window kept showing a loaded model with an Unload button
       * for a server that no longer existed — the app and the machine disagreeing about whether
       * anything was running.
       */
      const wasLoaded = this.current?.port === port
      if (wasLoaded) this.current = null
      this.modalityCache = null
      this.child = null

      /*
       * A server that dies leaves a record.
       *
       * Loads were logged and exits were not, so a crash left no trace whatsoever — the log
       * showed a model loading four times and never once stopping, and there was no way to find
       * out afterwards what had happened. The tail of stderr is where llama.cpp puts the reason,
       * an allocation failure above all.
       */
      if (wasLoaded || code !== 0) {
        logger.warn('model', `llama-server exited (code ${code}, signal ${signal ?? 'none'})`, {
          model: model.filename,
          contextLength: plan.contextLength,
          stderr: stderr.slice(-4000)
        })
      }

      this.emit('status', { phase: 'exited', code })
    })

    try {
      await this.waitForHealth(port, child, () => stderr, () => spawnFailure)
    } catch (err) {
      // A load that fails takes its child with it. Left running, a server that came up but never
      // reported healthy would sit on the model's VRAM until the next load happened to unload it
      // — or forever, if the user gave up and tried something else.
      await this.unload()
      throw err
    }

    /*
     * A projector that failed to allocate is worse than one that failed to load.
     *
     * These messages appear during startup and are not fatal to llama.cpp, so the health check
     * passes and everything looks correct. The first image is what finds out — by which point
     * the failure is an access violation rather than an error anyone can act on.
     */
    const allocationFailure = /cudaMalloc failed: out of memory|failed to allocate .*buffer/i.exec(stderr)
    const visionUnavailable =
      model.caps.mmprojPath && allocationFailure
        ? 'The vision projector could not reserve the memory it needs at this context length, so images and video would crash the model. Load with a smaller context to use them.'
        : undefined
    if (visionUnavailable) {
      logger.warn('model', 'vision projector failed to allocate; media will be refused', {
        model: model.filename,
        contextLength: plan.contextLength,
        detail: allocationFailure?.[0]
      })
    }

    const loaded: LoadedModel = { model, plan, port, startedAt: Date.now(), visionUnavailable }
    this.current = loaded
    this.modalityCache = null
    this.emit('status', { phase: 'ready', model: model.filename, port })
    return loaded
  }

  private async waitForHealth(
    port: number,
    child: ChildProcess,
    stderr: () => string,
    spawnFailure: () => Error | null = () => null
  ): Promise<void> {
    const deadline = Date.now() + 10 * 60 * 1000 // big models genuinely take minutes
    while (Date.now() < deadline) {
      const failed = spawnFailure()
      if (failed) {
        throw new Error(`llama-server could not be started: ${failed.message}\n${stderr().slice(-2000)}`)
      }
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
    this.modalityCache = null
      return
    }
    this.child = null
    this.current = null
    this.modalityCache = null
    await new Promise<void>((resolve) => {
      // Cleared when the child goes quietly, so a normal unload does not leave a four-second
      // timer holding the event loop open behind it — which on quit is four seconds of the
      // process refusing to exit.
      const force = setTimeout(() => {
        try {
          child.kill('SIGKILL')
        } catch {
          /* already gone */
        }
        resolve()
      }, 4000)
      child.once('exit', () => {
        clearTimeout(force)
        resolve()
      })
      child.kill()
    })
  }

  private requestBody(opts: CompletionOptions, stream: boolean): Record<string, unknown> {
    const body: Record<string, unknown> = {
      messages: opts.messages,
      stream,
      temperature: opts.temperature ?? 0.7,
      top_p: opts.topP ?? 0.95,
      cache_prompt: true
    }
    /*
     * Progress frames are only useful while streaming, and only cost anything there.
     *
     * A server built before the field existed ignores it rather than rejecting the request, so
     * this needs no version check — the events simply never arrive and the UI never shows a bar.
     */
    if (stream) body.return_progress = true
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

    /*
     * Context accounting, kept apart from the timing counters above.
     *
     * `generatedTokens` includes reasoning, which `completionTokens` deliberately does not —
     * that one feeds tokens-per-second and time-to-first-token, where mixing in thinking would
     * change what those numbers mean. Context does not care why a token exists, only that it is
     * taking up room.
     */
    const maxContext = loaded.plan.contextLength
    let generatedTokens = 0
    let contextSentAt = 0
    /** At most twice a second: often enough to look live, rarely enough not to flood the bridge. */
    const CONTEXT_INTERVAL_MS = 500

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

    /** Indices already reported as in progress, so a new call is always announced at once. */
    const announcedCalls = new Set<number>()
    let lastPartialAt = 0
    /** Fast enough to look like typing, slow enough not to flood the bridge. */
    const PARTIAL_INTERVAL_MS = 120

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
            prompt_progress?: { total?: number; cache?: number; processed?: number; time_ms?: number }
          }
          try {
            json = JSON.parse(payload)
          } catch {
            continue // partial frame; the next chunk completes it
          }

          /*
           * Progress frames arrive before generation and carry no delta, so they are read
           * before the choice is looked at rather than inside it.
           *
           * `processed` is clamped to `total` because the server counts a cached prefix as
           * processed the moment it is reused, which can briefly overshoot on a follow-up turn.
           */
          const pp = json.prompt_progress
          if (pp && typeof pp.total === 'number' && pp.total > 0) {
            const processed = Math.max(0, Math.min(pp.processed ?? 0, pp.total))
            yield {
              type: 'prompt_progress',
              processed,
              total: pp.total,
              cached: pp.cache ?? 0,
              percent: Math.round((processed / pp.total) * 100)
            }
            /*
             * The prompt total is the context this turn starts from, and it is known here —
             * before a single token has been generated. Reporting it now means the reading is
             * right from the beginning of a turn rather than catching up at the end of one.
             */
            if (pp.total !== promptTokens) {
              promptTokens = pp.total
              contextSentAt = Date.now()
              yield { type: 'context', used: promptTokens + generatedTokens, max: maxContext }
            }
          }

          const choice = json.choices?.[0]
          const delta = choice?.delta

          if (delta?.reasoning_content) {
            // Thinking counts towards time-to-first-token: it is the model working, and a
            // reasoning model can spend a long time here before any answer appears.
            if (ttft === null) ttft = Date.now() - startedAt
            generatedTokens += estimateTokens(delta.reasoning_content)
            yield { type: 'reasoning', text: delta.reasoning_content }
            if (Date.now() - contextSentAt >= CONTEXT_INTERVAL_MS) {
              contextSentAt = Date.now()
              yield { type: 'context', used: promptTokens + generatedTokens, max: maxContext }
            }
          }

          if (delta?.content) {
            if (ttft === null) ttft = Date.now() - startedAt
            const n = estimateTokens(delta.content)
            completionTokens += n
            generatedTokens += n
            yield { type: 'text', text: delta.content }
            if (Date.now() - contextSentAt >= CONTEXT_INTERVAL_MS) {
              contextSentAt = Date.now()
              yield { type: 'context', used: promptTokens + generatedTokens, max: maxContext }
            }
          }

          if (delta?.tool_calls) {
            for (const frag of delta.tool_calls) {
              const idx = frag.index ?? 0
              const existing = pending.get(idx) ?? { id: frag.id ?? `call_${idx}`, name: '', args: '' }
              if (frag.id) existing.id = frag.id
              if (frag.function?.name) existing.name += frag.function.name
              if (frag.function?.arguments) existing.args += frag.function.arguments
              pending.set(idx, existing)

              /*
               * Report the call as it is being written, not only once it is finished.
               *
               * Immediately when a call first appears, because knowing *which* tool is about to
               * run is the most useful thing here and it is known from the first fragment. After
               * that on a throttle: arguments arrive a few characters at a time, and forwarding
               * every fragment would put hundreds of messages across the bridge to animate a
               * line of text.
               */
              const now = Date.now()
              const isNew = !announcedCalls.has(idx)
              if (isNew || now - lastPartialAt >= PARTIAL_INTERVAL_MS) {
                announcedCalls.add(idx)
                lastPartialAt = now
                yield { type: 'tool_call_partial', index: idx, name: existing.name, args: existing.args }
              }
            }
          }

          if (json.usage) {
            promptTokens = json.usage.prompt_tokens ?? promptTokens
            generatedTokens = json.usage.completion_tokens ?? generatedTokens
            yield {
              type: 'usage',
              promptTokens,
              completionTokens: json.usage.completion_tokens ?? completionTokens
            }
            contextSentAt = Date.now()
            yield { type: 'context', used: promptTokens + generatedTokens, max: maxContext }
          }

          if (choice?.finish_reason) {
            yield* flushCalls()
          }
        }
      }
      yield* flushCalls()
      // Prompt tokens carried through here too. A stream that ends by the body simply closing —
      // rather than with an explicit [DONE] — took the other branch, which omitted the argument
      // and reported prompt_tokens as zero to every client that reads usage.
      this.lastTimings = finalise(startedAt, ttft, completionTokens, promptTokens)
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

  /**
   * Like `complete`, but also surfaces tool calls and thinking instead of discarding them.
   *
   * `complete`/`stream` exist for plain-chat callers that only want text; API clients that pass
   * `tools` need the calls themselves, which only `streamEvents` carries.
   *
   * Reasoning is accumulated separately rather than folded into `text`. llama.cpp runs with
   * `--reasoning-format deepseek`, which strips thinking out of the answer and returns it as
   * `reasoning_content` — so appending it to the text would put the model's working inside its
   * reply, which is exactly what that flag exists to prevent. Callers that do not want it can
   * ignore the field; the one that dropped it had no way to offer it at all.
   */
  async completeFull(
    opts: CompletionOptions
  ): Promise<{ text: string; reasoning: string; toolCalls: StreamedToolCall[] }> {
    let text = ''
    let reasoning = ''
    const toolCalls: StreamedToolCall[] = []
    for await (const ev of this.streamEvents(opts)) {
      if (ev.type === 'text') text += ev.text
      else if (ev.type === 'reasoning') reasoning += ev.text
      else if (ev.type === 'tool_call') toolCalls.push(ev.call)
    }
    return { text, reasoning, toolCalls }
  }

  /** Embeddings via the same server; used by RAG when the embedding model is loaded. */
  async embed(texts: string[], port?: number): Promise<number[][]> {
    const target = port ?? this.current?.port
    if (!target) throw new Error('No model is loaded for embeddings')
    const res = await fetch(`http://127.0.0.1:${target}/v1/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input: texts }),
      // A server that has wedged would otherwise hang ingestion for the life of the process,
      // with no error and nothing to cancel.
      signal: AbortSignal.timeout(5 * 60 * 1000)
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
