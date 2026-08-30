/**
 * Logging and the diagnostics bundle.
 *
 * Local files only — no telemetry, ever, by decision. The "copy diagnostics" bundle exists so
 * a user can hand over something useful when reporting a bug, and it redacts secrets before
 * writing, since the whole point is that it gets shared.
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'
import { LOGS_DIR, APPDATA_DIR } from '../storage/paths'
import { loadSettings } from '../storage/settings'
import { detectHardware } from '../hardware/gpu'
import { llama } from '../runtime/llama'
import { missingBinaries } from '../runtime/binaries'
import { all } from '../storage/db'

const MAX_LOG_BYTES = 8 * 1024 * 1024
const KEEP_FILES = 5

let stream: fs.WriteStream | null = null
let currentFile: string | null = null

function logFile(): string {
  return path.join(LOGS_DIR, `llmmanager-${new Date().toISOString().slice(0, 10)}.log`)
}

/** Set while a rotation is closing the old file; writes are dropped rather than misdirected. */
let rotating = false
/** When rotation last failed, so a locked file is not retried on every single line. */
let rotationBlockedUntil = 0

/**
 * The open log stream, or null while the file is being rotated.
 *
 * Rotation used to be `end()` then `renameSync()` in one synchronous breath. On Windows a file
 * with an open write handle cannot be renamed, and `end()` has not closed the handle by the time
 * the next statement runs — so the rename threw, the catch swallowed it, and `stream` was left
 * pointing at a stream that had already been ended. Every later write then threw into the
 * catch-all in `log()`, and the app stopped logging entirely, silently, for the rest of the
 * session. The close is awaited through its callback instead, and nothing writes in between.
 */
function ensureStream(): fs.WriteStream | null {
  if (rotating) return null
  const target = logFile()

  if (stream && currentFile === target && Date.now() >= rotationBlockedUntil) {
    let size = 0
    try {
      size = fs.statSync(target).size
    } catch {
      size = 0
    }
    if (size > MAX_LOG_BYTES) {
      const finished = stream
      stream = null
      currentFile = null
      rotating = true
      finished.end(() => {
        try {
          fs.renameSync(target, `${target}.${Date.now()}`)
        } catch {
          // Something else is holding the file. Keep appending to it and try again in a while,
          // rather than attempting a rename per log line.
          rotationBlockedUntil = Date.now() + 60_000
        }
        rotating = false
        void pruneOldLogs()
      })
      return null
    }
  }

  if (!stream || currentFile !== target) {
    fs.mkdirSync(LOGS_DIR, { recursive: true })
    stream = fs.createWriteStream(target, { flags: 'a' })
    currentFile = target
    void pruneOldLogs()
  }
  return stream
}

async function pruneOldLogs(): Promise<void> {
  try {
    const files = (await fsp.readdir(LOGS_DIR))
      .filter((f) => f.startsWith('llmmanager-'))
      .map((f) => ({ f, full: path.join(LOGS_DIR, f) }))
    if (files.length <= KEEP_FILES) return
    const stats = await Promise.all(files.map(async (x) => ({ ...x, mtime: (await fsp.stat(x.full)).mtimeMs })))
    stats.sort((a, b) => b.mtime - a.mtime)
    for (const old of stats.slice(KEEP_FILES)) await fsp.rm(old.full, { force: true })
  } catch {
    /* best effort */
  }
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error'

export function log(level: LogLevel, scope: string, message: string, extra?: unknown): void {
  const line = `${new Date().toISOString()} ${level.toUpperCase().padEnd(5)} [${scope}] ${message}${
    extra !== undefined ? ` ${safeJson(extra)}` : ''
  }\n`
  try {
    // Null while a rotation is in flight — a handful of lines during the swap is the price of
    // never writing into a file that is about to be renamed out from under the handle.
    ensureStream()?.write(line)
  } catch {
    /* logging must never throw */
  }
  if (level === 'error' || level === 'warn') process.stderr.write(line)
}

function safeJson(v: unknown): string {
  try {
    return JSON.stringify(v)
  } catch {
    return String(v)
  }
}

export const logger = {
  debug: (scope: string, msg: string, extra?: unknown) => log('debug', scope, msg, extra),
  info: (scope: string, msg: string, extra?: unknown) => log('info', scope, msg, extra),
  warn: (scope: string, msg: string, extra?: unknown) => log('warn', scope, msg, extra),
  error: (scope: string, msg: string, extra?: unknown) => log('error', scope, msg, extra)
}

const REDACTIONS: [RegExp, string][] = [
  [/\b(hf_[A-Za-z0-9]{20,})\b/g, 'hf_***REDACTED***'],
  [/\b(llmm-[a-f0-9]{40,})\b/g, 'llmm-***REDACTED***'],
  [/("(?:password|apiKey|token|hfToken|sessionKey|passwordHash|passwordSalt)"\s*:\s*)"[^"]*"/gi, '$1"***REDACTED***"']
]

function redact(text: string): string {
  let out = text
  for (const [re, replacement] of REDACTIONS) out = out.replace(re, replacement)
  return out
}

/**
 * Build the diagnostics bundle: logs, hardware, settings (redacted), the loaded plan, and
 * recent tool activity. Written as one text file the user can paste or attach.
 */
export async function buildDiagnostics(): Promise<{ path: string; text: string }> {
  const hardware = await detectHardware().catch(() => null)
  const settings = loadSettings()
  const loaded = llama.loaded

  const sections: string[] = []

  sections.push(
    '=== LLM Manager diagnostics ===',
    `Generated: ${new Date().toISOString()}`,
    `App data:  ${APPDATA_DIR}`,
    ''
  )

  sections.push(
    '--- System ---',
    `OS:      ${os.type()} ${os.release()} (${os.arch()})`,
    `CPU:     ${hardware?.cpuName ?? 'unknown'} (${hardware?.cpuThreads ?? '?'} threads)`,
    `RAM:     ${fmt(hardware?.totalRam ?? 0)} total, ${fmt(hardware?.freeRam ?? 0)} free`,
    `Backend: ${hardware?.backend ?? 'unknown'}`,
    `Node:    ${process.versions.node}  Electron: ${process.versions.electron}`,
    ''
  )

  sections.push('--- GPUs ---')
  for (const g of hardware?.gpus ?? []) {
    sections.push(
      `${g.name} (${g.vendor}) — total ${fmt(g.totalVram)}, free ${g.freeVram >= 0 ? fmt(g.freeVram) : 'unmeasured'}, ` +
        `measured=${g.freeIsMeasured}, util=${g.utilisation}%`
    )
  }
  if (!hardware?.gpus.length) sections.push('(none detected)')
  sections.push('')

  sections.push('--- Missing bundled binaries ---')
  const missing = hardware ? missingBinaries(hardware.backend) : []
  sections.push(missing.length ? missing.join(', ') : '(none)', '')

  sections.push('--- Loaded model ---')
  if (loaded) {
    sections.push(
      `File:    ${loaded.model.filename}`,
      `Arch:    ${loaded.model.arch?.architecture ?? '?'} ${loaded.model.arch?.quant ?? ''}`,
      `Context: ${loaded.plan.contextLength}`,
      `KV:      ${loaded.plan.kvType}`,
      `Layers:  ${loaded.plan.gpuLayers}/${loaded.plan.totalLayers} on GPU`,
      `Split:   ${loaded.plan.tensorSplit.map((s) => s.toFixed(3)).join(', ')}`,
      `Predicted VRAM: ${loaded.plan.predictedVramPerGpu.map(fmt).join(' + ')}`
    )
  } else {
    sections.push('(no model loaded)')
  }
  sections.push('')

  sections.push('--- Settings (redacted) ---', redact(JSON.stringify(settings, null, 2)), '')

  sections.push('--- Recent API requests ---')
  try {
    const rows = all<{ ts: number; endpoint: string; ms: number; client: string; status: number }>(
      'SELECT ts, endpoint, ms, client, status FROM requests ORDER BY ts DESC LIMIT 25'
    )
    for (const r of rows) {
      sections.push(`${new Date(r.ts).toISOString()}  ${r.endpoint}  ${r.ms}ms  ${r.client}  ${r.status}`)
    }
    if (!rows.length) sections.push('(none)')
  } catch {
    sections.push('(request log unavailable)')
  }
  sections.push('')

  sections.push('--- Log tail ---')
  try {
    const file = logFile()
    const content = await fsp.readFile(file, 'utf8')
    sections.push(redact(content.slice(-40000)))
  } catch {
    sections.push('(no log file yet)')
  }

  const text = sections.join('\n')
  const outPath = path.join(APPDATA_DIR, `diagnostics-${Date.now()}.txt`)
  await fsp.writeFile(outPath, text, 'utf8')
  return { path: outPath, text }
}

function fmt(n: number): string {
  if (!n) return '0'
  const gb = n / 1024 ** 3
  return gb >= 1 ? `${gb.toFixed(2)} GB` : `${(n / 1024 ** 2).toFixed(0)} MB`
}

/** Route uncaught failures into the log rather than losing them. */
export function installCrashHandlers(): void {
  process.on('uncaughtException', (err) => {
    logger.error('uncaught', err.message, err.stack)
  })
  process.on('unhandledRejection', (reason) => {
    logger.error('unhandled-rejection', String(reason))
  })
}
