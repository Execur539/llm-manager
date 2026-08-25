/**
 * Execution tools: shell, background processes, and the bundled interpreters.
 *
 * Background processes are the reason this is not just `exec`: a build or a test run should
 * not block the agent loop, and its output needs to be readable in pieces afterwards.
 */

import { spawn, ChildProcess } from 'node:child_process'
import path from 'node:path'
import fs from 'node:fs'
import os from 'node:os'
import crypto from 'node:crypto'
import type { Tool, ToolContext } from './base'
import { schema, str, int, bool } from './base'
import { runtimeBinary } from '../../runtime/binaries'

interface BackgroundJob {
  id: string
  command: string
  child: ChildProcess
  output: string[]
  exitCode: number | null
  startedAt: number
  /** set when the process closes; drives reaping */
  exitedAt: number | null
}

const jobs = new Map<string, BackgroundJob>()

/**
 * Finished jobs are kept around so the agent can still read their output, but not forever —
 * each one pins a ChildProcess and up to 4000 chunks of buffered output, and a long session
 * that builds in a loop would otherwise grow this map without bound. Running jobs are never
 * reaped; only completed ones age out.
 */
const FINISHED_TTL_MS = 30 * 60 * 1000
const MAX_FINISHED = 50

function reapJobs(): void {
  const finished = [...jobs.values()]
    .filter((j) => j.exitedAt !== null)
    .sort((a, b) => (a.exitedAt as number) - (b.exitedAt as number))

  const cutoff = Date.now() - FINISHED_TTL_MS
  const excess = Math.max(0, finished.length - MAX_FINISHED)

  finished.forEach((job, i) => {
    if (i < excess || (job.exitedAt as number) < cutoff) jobs.delete(job.id)
  })
}

function shellFor(command: string): { file: string; args: string[] } {
  // PowerShell is the primary shell on this platform; -Command handles pipelines and
  // multi-line input without needing a temp script.
  return {
    file: 'powershell.exe',
    args: ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', command]
  }
}

function runToCompletion(
  file: string,
  args: string[],
  cwd: string,
  timeoutMs: number,
  signal: AbortSignal,
  env?: NodeJS.ProcessEnv
): Promise<{ stdout: string; stderr: string; code: number | null; timedOut: boolean }> {
  return new Promise((resolve) => {
    const child = spawn(file, args, { cwd, windowsHide: true, env: { ...process.env, ...env } })
    let stdout = ''
    let stderr = ''
    let timedOut = false
    const MAX = 8 * 1024 * 1024

    const timer = setTimeout(() => {
      timedOut = true
      child.kill('SIGKILL')
    }, timeoutMs)

    const onAbort = (): void => {
      child.kill("SIGKILL")
    }
    // A listener attached to an already-aborted signal never fires, so a command dispatched
    // just after the user hit stop would otherwise run to completion unsupervised.
    if (signal.aborted) onAbort()
    else signal.addEventListener('abort', onAbort, { once: true })

    child.stdout?.on('data', (d: Buffer) => {
      if (stdout.length < MAX) stdout += d.toString()
    })
    child.stderr?.on('data', (d: Buffer) => {
      if (stderr.length < MAX) stderr += d.toString()
    })
    child.on('error', (err) => {
      stderr += `\n${err.message}`
    })
    child.on('close', (code) => {
      clearTimeout(timer)
      signal.removeEventListener('abort', onAbort)
      resolve({ stdout, stderr, code, timedOut })
    })
  })
}

function formatRun(r: { stdout: string; stderr: string; code: number | null; timedOut: boolean }): string {
  const parts: string[] = []
  if (r.stdout.trim()) parts.push(r.stdout.trimEnd())
  if (r.stderr.trim()) parts.push(`[stderr]\n${r.stderr.trimEnd()}`)
  if (r.timedOut) parts.push('[timed out and was killed]')
  parts.push(`[exit code ${r.code ?? 'null'}]`)
  return parts.join('\n')
}

const runCommand: Tool = {
  name: 'run_command',
  description:
    'Run a shell command (PowerShell) and wait for it to finish. Use run_background for anything ' +
    'long-running like a dev server, build watch, or test suite you want to keep going.',
  tier: 'execute',
  parameters: schema(
    {
      command: str('The command to run'),
      cwd: str('Working directory (default: session working directory)'),
      timeout_ms: int('Kill the command after this many milliseconds')
    },
    ['command']
  ),
  async run(args, ctx) {
    const cwd = args.cwd ? path.resolve(ctx.cwd, String(args.cwd)) : ctx.cwd
    const { file, args: shellArgs } = shellFor(String(args.command))
    const timeout = Number(args.timeout_ms ?? ctx.timeoutMs)
    return formatRun(await runToCompletion(file, shellArgs, cwd, timeout, ctx.signal))
  }
}

const runBackground: Tool = {
  name: 'run_background',
  description:
    'Start a long-running command in the background and return a job id immediately. ' +
    'Read its output with read_job and stop it with kill_job.',
  tier: 'execute',
  parameters: schema({ command: str('Command to run'), cwd: str('Working directory') }, ['command']),
  async run(args, ctx) {
    const cwd = args.cwd ? path.resolve(ctx.cwd, String(args.cwd)) : ctx.cwd
    const { file, args: shellArgs } = shellFor(String(args.command))
    const id = crypto.randomBytes(4).toString('hex')

    const child = spawn(file, shellArgs, { cwd, windowsHide: true, detached: false })
    const job: BackgroundJob = {
      id,
      command: String(args.command),
      child,
      output: [],
      exitCode: null,
      startedAt: Date.now(),
      exitedAt: null
    }
    const push = (d: Buffer): void => {
      job.output.push(d.toString())
      // Keep memory bounded on a chatty process.
      if (job.output.length > 4000) job.output.splice(0, job.output.length - 4000)
    }
    child.stdout?.on('data', push)
    child.stderr?.on('data', push)
    child.on('close', (code) => {
      job.exitCode = code
      job.exitedAt = Date.now()
    })

    reapJobs()
    jobs.set(id, job)
    return `Started background job ${id}: ${job.command}`
  }
}

const readJob: Tool = {
  name: 'read_job',
  description: 'Read accumulated output from a background job started with run_background.',
  tier: 'read',
  parameters: schema({ job_id: str('Job id'), clear: bool('Clear the buffer after reading') }, ['job_id']),
  async run(args) {
    const job = jobs.get(String(args.job_id))
    if (!job) throw new Error(`No background job ${args.job_id}`)
    const text = job.output.join('')
    if (args.clear) job.output = []
    const status = job.exitCode === null ? 'running' : `exited (${job.exitCode})`
    return `[job ${job.id} ${status}, ${((Date.now() - job.startedAt) / 1000).toFixed(1)}s]\n${text || '(no output yet)'}`
  }
}

const killJob: Tool = {
  name: 'kill_job',
  description: 'Terminate a background job.',
  tier: 'execute',
  parameters: schema({ job_id: str('Job id') }, ['job_id']),
  async run(args) {
    const job = jobs.get(String(args.job_id))
    if (!job) throw new Error(`No background job ${args.job_id}`)
    job.child.kill('SIGKILL')
    return `Killed job ${job.id}`
  }
}

const listJobs: Tool = {
  name: 'list_jobs',
  description: 'List background jobs and their status.',
  tier: 'read',
  parameters: schema({}),
  async run() {
    if (!jobs.size) return 'No background jobs.'
    return [...jobs.values()]
      .map(
        (j) =>
          `${j.id}  ${j.exitCode === null ? 'running' : `exit ${j.exitCode}`}  ${((Date.now() - j.startedAt) / 1000).toFixed(0)}s  ${j.command}`
      )
      .join('\n')
  }
}

/**
 * Python comes from the bundled embeddable distribution so it exists on every install,
 * falling back to whatever is on PATH during development.
 */
async function runInterpreter(
  kind: 'python' | 'node',
  code: string,
  ctx: ToolContext
): Promise<string> {
  const tmp = path.join(os.tmpdir(), `llmm-${crypto.randomBytes(6).toString('hex')}.${kind === 'python' ? 'py' : 'mjs'}`)
  fs.writeFileSync(tmp, code, 'utf8')
  try {
    const bin = kind === 'python' ? runtimeBinary('python') : runtimeBinary('node')
    const r = await runToCompletion(bin, [tmp], ctx.cwd, ctx.timeoutMs, ctx.signal)
    return formatRun(r)
  } finally {
    try {
      fs.unlinkSync(tmp)
    } catch {
      /* best effort */
    }
  }
}

const runPython: Tool = {
  name: 'run_python',
  description: 'Execute a Python script and return its output. Uses the interpreter bundled with the app.',
  tier: 'execute',
  parameters: schema({ code: str('Python source to execute') }, ['code']),
  async run(args, ctx) {
    return runInterpreter('python', String(args.code), ctx)
  }
}

const runNode: Tool = {
  name: 'run_node',
  description: 'Execute a JavaScript (ES module) script with Node and return its output.',
  tier: 'execute',
  parameters: schema({ code: str('JavaScript source to execute') }, ['code']),
  async run(args, ctx) {
    return runInterpreter('node', String(args.code), ctx)
  }
}

export const execTools: Tool[] = [runCommand, runBackground, readJob, killJob, listJobs, runPython, runNode]

/** Kill every background job — called on app quit so nothing is orphaned. */
export function killAllJobs(): void {
  for (const job of jobs.values()) {
    try {
      job.child.kill('SIGKILL')
    } catch {
      /* best effort */
    }
  }
  jobs.clear()
}
