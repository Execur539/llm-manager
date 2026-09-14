/**
 * A second llama-server for compaction, running a small model beside the loaded one.
 *
 * Started the first time a compaction asks for it, and stopped whenever the loaded model changes:
 * where it fits depends on what the main model left free, so a placement chosen against one model
 * is not safe to keep against the next.
 */

import { spawn, type ChildProcess } from 'node:child_process'
import fs from 'node:fs'
import net from 'node:net'
import type { Backend, ModelRecord } from '@shared/types'
import { childEnv, llamaServerPath } from './binaries'
import { logger } from '../log'
import { HELPER_UBATCH, helperContext, type HelperPlacement } from '../agent/compaction'

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

export interface SummarizerHandle {
  /** What to call it in the interface. */
  label: string
  /** The helper's window, which chunk sizes are planned against. */
  contextTokens: number
  placement: HelperPlacement
  complete(system: string, user: string, maxTokens: number, signal?: AbortSignal): Promise<string>
}

class SummarizerService {
  private child: ChildProcess | null = null
  private key: string | null = null
  private handle: SummarizerHandle | null = null
  private starting: Promise<SummarizerHandle> | null = null
  /** Bumped by every stop, so a start that was overtaken by one does not come back to life. */
  private generation = 0

  async ensure(model: ModelRecord, placement: HelperPlacement, backend: Backend): Promise<SummarizerHandle> {
    const key = `${model.path}|${placement.kind === 'gpu' ? `cuda${placement.index}` : 'cpu'}`
    if (this.handle && this.child && this.key === key) return this.handle
    if (this.starting) {
      await this.starting.catch(() => undefined)
      if (this.handle && this.child && this.key === key) return this.handle
    }
    this.starting = this.start(model, placement, backend, key).finally(() => {
      this.starting = null
    })
    return this.starting
  }

  private async start(
    model: ModelRecord,
    placement: HelperPlacement,
    backend: Backend,
    key: string
  ): Promise<SummarizerHandle> {
    await this.stop()
    const generation = this.generation
    if (!fs.existsSync(model.path)) throw new Error(`Summariser model not found at ${model.path}`)

    const useMock = process.env.LLMM_MOCK_LLAMA === '1'
    /*
     * The CPU build for a helper on the CPU.
     *
     * The CUDA build with no layers offloaded still opens a context on every GPU it can see, which
     * is a few hundred megabytes taken from the loaded model on each card for a process that never
     * uses them. GPU placement is CUDA-only: the index comes from nvidia-smi, which only CUDA's
     * numbering is pinned to.
     */
    const onGpu = placement.kind === 'gpu' && backend === 'cuda'
    const exe = useMock ? process.execPath : llamaServerPath(onGpu ? 'cuda' : 'cpu')
    const contextTokens = helperContext(model.arch)
    const port = await freePort()
    const args = [
      ...(useMock ? [process.env.LLMM_MOCK_SCRIPT ?? ''] : []),
      '--model', model.path,
      '--host', '127.0.0.1',
      '--port', String(port),
      '--ctx-size', String(contextTokens),
      '--parallel', '1',
      '--jinja',
      '--flash-attn', 'on',
      '--cache-type-k', 'q8_0',
      '--cache-type-v', 'q8_0',
      // A summariser that reasoned first would spend the time it exists to save.
      '--reasoning', 'off',
      '--ubatch-size', String(HELPER_UBATCH),
      ...(onGpu && placement.kind === 'gpu'
        ? ['--n-gpu-layers', 'all', '--device', `CUDA${placement.index}`]
        : ['--n-gpu-layers', '0'])
    ]

    const child = spawn(exe, args, {
      windowsHide: true,
      env: useMock ? { ...childEnv(), ELECTRON_RUN_AS_NODE: '1' } : childEnv()
    })
    this.child = child

    // Read, not just opened: an unread pipe fills and blocks the server on its next log line.
    let log = ''
    const capture = (d: Buffer): void => {
      log = `${log}${d.toString()}`.slice(-8192)
    }
    child.stdout?.on('data', capture)
    child.stderr?.on('data', capture)
    child.on('exit', () => {
      if (this.child === child) {
        this.child = null
        this.handle = null
        this.key = null
      }
    })

    let healthy = false
    const deadline = Date.now() + 120_000
    while (Date.now() < deadline) {
      if (child.exitCode !== null) throw new Error(`Summariser exited (${child.exitCode})\n${log.slice(-1000)}`)
      if (generation !== this.generation) break
      healthy = await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(1500) })
        .then((r) => r.ok)
        .catch(() => false)
      if (healthy) break
      await new Promise((r) => setTimeout(r, 400))
    }
    if (generation !== this.generation) {
      child.kill()
      throw new Error('Summariser was stopped while it started')
    }
    if (!healthy) {
      await this.stop()
      throw new Error(`Summariser did not become ready within two minutes.\n${log.slice(-1000)}`)
    }

    const label = model.arch?.name?.trim() || model.filename.replace(/\.gguf$/i, '')
    const handle: SummarizerHandle = {
      label,
      contextTokens,
      placement: onGpu ? placement : { kind: 'cpu', reason: placement.kind === 'cpu' ? placement.reason : 'GPU placement needs CUDA' },
      complete: async (system, user, maxTokens, signal) => {
        const res = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: [
              { role: 'system', content: system },
              { role: 'user', content: user }
            ],
            temperature: 0.2,
            max_tokens: maxTokens,
            // Qwen's own guidance for its small models, whose failure mode is repeating themselves.
            presence_penalty: 1.5,
            stream: false
          }),
          signal
        })
        if (!res.ok) throw new Error(`Summariser request failed: HTTP ${res.status}`)
        const json = (await res.json()) as { choices?: { message?: { content?: string } }[] }
        return (json.choices?.[0]?.message?.content ?? '').trim()
      }
    }
    this.key = key
    this.handle = handle
    logger.info('compaction', `summariser ${label} ready`, {
      placement: onGpu && placement.kind === 'gpu' ? `CUDA${placement.index} (${placement.name})` : 'CPU',
      contextTokens
    })
    return handle
  }

  async stop(): Promise<void> {
    this.generation++
    const child = this.child
    this.child = null
    this.handle = null
    this.key = null
    if (!child || child.exitCode !== null) return
    await new Promise<void>((resolve) => {
      const timer = setTimeout(resolve, 3000)
      child.once('exit', () => {
        clearTimeout(timer)
        resolve()
      })
      child.kill()
    })
  }
}

export const summarizer = new SummarizerService()
