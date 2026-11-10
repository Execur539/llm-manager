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

/** Where the agent's own throwaway scripts go, so they do not pile up in the user's folders. */
export const AGENT_SCRATCH_DIR = path.join(os.tmpdir(), 'LLMManager-agent')

/** The scratch folder, created on first use and cleared of anything more than a week old. */
let scratchReady = false
export function agentScratchDir(): string {
  if (!scratchReady) {
    scratchReady = true
    try {
      fs.mkdirSync(AGENT_SCRATCH_DIR, { recursive: true })
      const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000
      for (const name of fs.readdirSync(AGENT_SCRATCH_DIR)) {
        const file = path.join(AGENT_SCRATCH_DIR, name)
        const stat = fs.statSync(file)
        if (stat.isFile() && stat.mtimeMs < cutoff) fs.rmSync(file, { force: true })
      }
    } catch {
      /* a scratch folder that cannot be swept is still usable */
    }
  }
  return AGENT_SCRATCH_DIR
}

/**
 * Run before every command: output as UTF-8, and no progress bars.
 *
 * Windows PowerShell writes to a pipe in the console's legacy code page, so anything outside ASCII
 * — a file name, a degree sign, a non-English event message — reached the model as replacement
 * characters. On the same line as the start of the command, so the line numbers in an error
 * message are the command's own.
 */
const PS_PRELUDE =
  "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false; $OutputEncoding = [Console]::OutputEncoding; $ProgressPreference = 'SilentlyContinue'; "

/**
 * Run after every command: fail when its last statement failed.
 *
 * That is what `-Command` reported, and a script file reports success unless it calls `exit`
 * itself. Without this a command that ended in an error would come back as exit code 0.
 */
const PS_EPILOGUE = '\r\nif (-not $?) { if ($LASTEXITCODE) { exit $LASTEXITCODE } else { exit 1 } }\r\n'

/**
 * How a command reaches PowerShell: as a script file in the scratch folder.
 *
 * Not as `-Command <text>`, which sends the script through Windows' own command-line quoting. That
 * mangles nested double quotes, and the model's way round it was to write a .ps1 into the user's
 * folders and run that instead. Not as `-EncodedCommand` either: it survives quoting, but
 * PowerShell then writes every error and progress record to the error stream as XML. A file is
 * read exactly as written, at any length, and errors come back as text.
 *
 * `cleanup` removes the file once the command has finished.
 */
function shellFor(command: string): { file: string; args: string[]; cleanup: () => void } {
  const script = path.join(agentScratchDir(), `command-${crypto.randomBytes(6).toString('hex')}.ps1`)
  // With a byte-order mark, which is how Windows PowerShell knows a script file is UTF-8.
  fs.writeFileSync(script, `\uFEFF${PS_PRELUDE}${command}${PS_EPILOGUE}`, 'utf8')
  return {
    file: 'powershell.exe',
    args: ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', script],
    cleanup: () => fs.rmSync(script, { force: true })
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

/**
 * Run a PowerShell script for another tool and hand back what it printed.
 *
 * Through the same script-file path as run_command, so a tool that drives Windows from PowerShell
 * gets the same guarantees: the script arrives exactly as written and its output is UTF-8.
 * `env` carries the tool's arguments, which keeps them out of the script text and so out of
 * PowerShell's quoting rules altogether.
 */
export async function runPowerShell(
  script: string,
  opts: { timeoutMs: number; signal: AbortSignal; env?: NodeJS.ProcessEnv; cwd?: string }
): Promise<{ stdout: string; stderr: string; code: number | null; timedOut: boolean }> {
  const { file, args, cleanup } = shellFor(script)
  try {
    return await runToCompletion(file, args, opts.cwd ?? os.tmpdir(), opts.timeoutMs, opts.signal, opts.env)
  } finally {
    cleanup()
  }
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
    'Run a command or a whole multi-line script in Windows PowerShell 5.1 and wait for it to finish. ' +
    'Pass the script itself: there is no need to wrap it in another powershell call or save it to a ' +
    'file first. PowerShell 7 syntax (&&, ||, ??, the ternary operator) is not available. Use ' +
    'run_background for anything long-running like a dev server, build watch, or test suite you ' +
    'want to keep going, and run_elevated for anything that needs administrator rights.',
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
    const { file, args: shellArgs, cleanup } = shellFor(String(args.command))
    const timeout = Number(args.timeout_ms ?? ctx.timeoutMs)
    try {
      return formatRun(await runToCompletion(file, shellArgs, cwd, timeout, ctx.signal))
    } finally {
      cleanup()
    }
  }
}

const runBackground: Tool = {
  name: 'run_background',
  description:
    'Start a long-running command in the background and return a job id immediately. ' +
    'Read its output with read_job, answer anything it asks with write_job, and stop it with ' +
    'kill_job. Use this for a program that prompts for input.',
  tier: 'execute',
  parameters: schema({ command: str('Command to run'), cwd: str('Working directory') }, ['command']),
  async run(args, ctx) {
    const cwd = args.cwd ? path.resolve(ctx.cwd, String(args.cwd)) : ctx.cwd
    const { file, args: shellArgs, cleanup } = shellFor(String(args.command))
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
      cleanup()
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

/**
 * Type into a running job.
 *
 * A job could be started, read and killed, but nothing could answer it — so an installer asking
 * "continue? [y/n]", a sign-in prompt or any interactive program sat waiting until it was killed.
 */
const writeJob: Tool = {
  name: 'write_job',
  description:
    'Send text to the input of a background job started with run_background, as if typed at its ' +
    'prompt. Enter is pressed afterwards unless enter is false. Read the reply with read_job.',
  tier: 'execute',
  parameters: schema(
    {
      job_id: str('Job id'),
      text: str('What to type'),
      enter: bool('Press Enter after the text (default true)'),
      close_input: bool('Close the input afterwards, for a program that reads until end of input')
    },
    ['job_id', 'text']
  ),
  async run(args) {
    const job = jobs.get(String(args.job_id))
    if (!job) throw new Error(`No background job ${args.job_id}`)
    if (job.exitCode !== null) return `Job ${job.id} has already exited (${job.exitCode}); there is nothing to type into.`
    const stdin = job.child.stdin
    if (!stdin || stdin.destroyed || !stdin.writable) return `Job ${job.id} is not accepting input.`
    const text = String(args.text ?? '') + (args.enter === false ? '' : '\r\n')
    await new Promise<void>((resolve, reject) => stdin.write(text, (err) => (err ? reject(err) : resolve())))
    if (args.close_input) stdin.end()
    // Long enough for a prompt to answer, so the usual next step — reading the reply — is already here.
    await new Promise((r) => setTimeout(r, 400))
    const tail = job.output.join('').slice(-1500)
    return `Sent to job ${job.id}.${tail ? `\nOutput so far ends with:\n${tail}` : ''}`
  }
}

/** End a job and everything it started: the shell is only the parent of the program that matters. */
function killTree(child: ChildProcess): void {
  if (process.platform === 'win32' && child.pid) {
    spawn('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' }).on(
      'error',
      () => child.kill('SIGKILL')
    )
    return
  }
  child.kill('SIGKILL')
}

const killJob: Tool = {
  name: 'kill_job',
  description: 'Terminate a background job and any programs it started.',
  tier: 'execute',
  parameters: schema({ job_id: str('Job id') }, ['job_id']),
  async run(args) {
    const job = jobs.get(String(args.job_id))
    if (!job) throw new Error(`No background job ${args.job_id}`)
    killTree(job.child)
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
    /*
     * `node` resolves to Electron's own binary, which is only Node when told to be.
     *
     * Without ELECTRON_RUN_AS_NODE, `electron script.mjs` treats the script as an application
     * entry point: the code does run and its output does arrive, but the process then sits
     * waiting for an app lifecycle that never begins and never exits. `runToCompletion` waits
     * for `close`, so every run_node call blocked for the full command timeout — two minutes by
     * default — and came back as "[timed out and was killed]". The agent read that as failure
     * and retried or gave up, on a script that had worked.
     */
    const env = kind === 'node' ? { ELECTRON_RUN_AS_NODE: '1' } : undefined
    const r = await runToCompletion(bin, [tmp], ctx.cwd, ctx.timeoutMs, ctx.signal, env)
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

/**
 * Run a script as administrator and hand back what it printed.
 *
 * An elevated process cannot share this one's pipes, so its output is written to a file and read
 * back. Done by hand that is four steps — write a script, start it with RunAs, wait, read the log
 * — and the model got one of them wrong often enough to be worth doing here. Windows shows its
 * own consent prompt; declining it comes back as a refusal the model can read.
 */
const runElevated: Tool = {
  name: 'run_elevated',
  description:
    'Run a Windows PowerShell 5.1 command or script as administrator and wait for it to finish. ' +
    'Windows asks the user to approve with its own prompt. Returns everything the script printed. ' +
    'Use only when a command fails for lack of administrator rights.',
  tier: 'execute',
  parameters: schema(
    {
      command: str('The command or script to run as administrator'),
      timeout_ms: int('Give up waiting after this many milliseconds')
    },
    ['command']
  ),
  async run(args, ctx) {
    const dir = agentScratchDir()
    const id = crypto.randomBytes(6).toString('hex')
    const script = path.join(dir, `elevated-${id}.ps1`)
    const output = path.join(dir, `elevated-${id}.log`)
    const quote = (p: string): string => `'${p.replace(/'/g, "''")}'`
    fs.writeFileSync(
      script,
      `\uFEFF$ProgressPreference = 'SilentlyContinue'\r\n& {\r\n${String(args.command)}\r\n} *>&1 | Out-File -FilePath ${quote(output)} -Encoding utf8 -Width 4096\r\n`,
      'utf8'
    )
    const launcher =
      `try { $p = Start-Process powershell.exe -Verb RunAs -WindowStyle Hidden -Wait -PassThru -ArgumentList ` +
      `'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',${quote(`"${script}"`)}; ` +
      `Write-Output "[elevated exit code $($p.ExitCode)]" } catch { Write-Output "[not run] $($_.Exception.Message)"; exit 1 }`
    const { file, args: shellArgs, cleanup } = shellFor(launcher)
    const timeout = Number(args.timeout_ms ?? ctx.timeoutMs)
    try {
      const run = await runToCompletion(file, shellArgs, ctx.cwd, timeout, ctx.signal)
      let printed = ''
      try {
        printed = fs.readFileSync(output, 'utf8').replace(/^\uFEFF/, '').trimEnd()
      } catch {
        /* nothing was written: the prompt was declined, or the script printed nothing */
      }
      return [printed, run.stdout.trim(), run.timedOut ? '[timed out waiting for the elevated script]' : '']
        .filter(Boolean)
        .join('\n')
    } finally {
      cleanup()
      fs.rmSync(script, { force: true })
      fs.rmSync(output, { force: true })
    }
  }
}

export const execTools: Tool[] = [
  runCommand,
  runElevated,
  runBackground,
  readJob,
  writeJob,
  killJob,
  listJobs,
  runPython,
  runNode
]

/** Kill every background job — called on app quit so nothing is orphaned. */
export function killAllJobs(): void {
  for (const job of jobs.values()) {
    try {
      if (job.exitCode === null) killTree(job.child)
      else job.child.kill('SIGKILL')
    } catch {
      /* best effort */
    }
  }
  jobs.clear()
}
