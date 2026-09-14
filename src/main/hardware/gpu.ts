/**
 * GPU / system detection.
 *
 * Design rule from the plan: nothing about the user's hardware may be assumed or hardcoded.
 * Everything here is probed at runtime, and every value carries whether it was *measured*
 * or *estimated* so the auto-fit engine can be honest about its confidence.
 *
 * Free VRAM is the number that matters (P1 failure mode: other apps size against total).
 * NVIDIA gives it to us directly via nvidia-smi. AMD/Intel have no equivalent CLI on
 * Windows, so we mark freeIsMeasured=false and the engine applies a wider safety margin.
 */

import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import os from 'node:os'
import type { Backend, GpuDevice, HardwareSnapshot } from '@shared/types'

const exec = promisify(execFile)

const MB = 1024 * 1024

async function run(cmd: string, args: string[], timeoutMs = 8000): Promise<string | null> {
  try {
    const { stdout } = await exec(cmd, args, { timeout: timeoutMs, windowsHide: true })
    return stdout
  } catch {
    return null
  }
}

async function powershell(script: string, timeoutMs = 12000): Promise<string | null> {
  return run(
    'powershell.exe',
    ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', script],
    timeoutMs
  )
}

/** What running a probe said: an answer, a missing tool, or a tool that did not answer. */
export type ProbeOutcome = { status: 'ok'; stdout: string } | { status: 'absent' } | { status: 'failed'; reason: string }

/**
 * Tell "not installed" apart from "did not answer".
 *
 * Both used to come back as null, so an nvidia-smi that timed out while the driver was busy read
 * exactly like a machine with no NVIDIA driver at all — and the app went on to plan every model for
 * the CPU for the rest of the session. A missing executable is an answer; a timeout, a crash or a
 * non-zero exit is not.
 */
export function classifyProbeError(err: unknown): ProbeOutcome {
  const e = (err ?? {}) as { code?: string | number; killed?: boolean; signal?: string | null; message?: string }
  if (e.code === 'ENOENT') return { status: 'absent' }
  if (e.killed || e.signal === 'SIGTERM' || e.code === 'ETIMEDOUT') return { status: 'failed', reason: 'timed out' }
  if (typeof e.code === 'number') return { status: 'failed', reason: `exited with code ${e.code}` }
  const first = typeof e.message === 'string' ? e.message.split('\n')[0].trim().slice(0, 160) : ''
  return { status: 'failed', reason: first || 'failed' }
}

async function probe(cmd: string, args: string[], timeoutMs = 8000): Promise<ProbeOutcome> {
  try {
    const { stdout } = await exec(cmd, args, { timeout: timeoutMs, windowsHide: true })
    return { status: 'ok', stdout }
  } catch (err) {
    return classifyProbeError(err)
  }
}

/** nvidia-smi's CSV, one device per line. */
export function parseNvidiaCsv(out: string): GpuDevice[] {
  const gpus: GpuDevice[] = []
  for (const line of out.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed) continue
    const parts = trimmed.split(',').map((s) => s.trim())
    if (parts.length < 5) continue
    const [index, name, total, free, util] = parts
    const totalMb = Number(total)
    const freeMb = Number(free)
    if (!Number.isFinite(totalMb) || !Number.isFinite(freeMb)) continue
    gpus.push({
      index: Number(index) || gpus.length,
      name,
      vendor: 'nvidia',
      totalVram: totalMb * MB,
      freeVram: freeMb * MB,
      utilisation: Number.isFinite(Number(util)) ? Number(util) : -1,
      freeIsMeasured: true
    })
  }
  return gpus
}

/** NVIDIA: exact free/total VRAM and utilisation, and whether nvidia-smi answered at all. */
async function detectNvidia(): Promise<{ gpus: GpuDevice[]; outcome: ProbeOutcome }> {
  const outcome = await probe('nvidia-smi', [
    '--query-gpu=index,name,memory.total,memory.free,utilization.gpu',
    '--format=csv,noheader,nounits'
  ])
  if (outcome.status !== 'ok') return { gpus: [], outcome }
  const gpus = parseNvidiaCsv(outcome.stdout)
  // Output that parses to nothing is not an answer either.
  if (!gpus.length && outcome.stdout.trim()) {
    return { gpus, outcome: { status: 'failed', reason: 'returned output that could not be read' } }
  }
  return { gpus, outcome }
}

/**
 * Vendor-neutral fallback via WMI + the driver registry key.
 *
 * Win32_VideoController.AdapterRAM is a uint32 and therefore lies about anything over 4 GB,
 * so we prefer HardwareInformation.qwMemorySize from the display class registry key, which
 * is a true 64-bit value.
 */
async function detectGeneric(): Promise<{ gpus: GpuDevice[]; ok: boolean }> {
  // WMI is Windows-only. Elsewhere there is nothing to ask, which is an answer rather than a failure.
  if (process.platform !== 'win32') return { gpus: [], ok: true }
  const script = `
$ErrorActionPreference='SilentlyContinue'
$class='HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}'
$out=@()
Get-CimInstance Win32_VideoController | ForEach-Object {
  $name=$_.Name
  $ram=[int64]$_.AdapterRAM
  $qw=0
  Get-ChildItem $class -ErrorAction SilentlyContinue | ForEach-Object {
    $p=Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue
    if ($p.'DriverDesc' -eq $name -and $p.'HardwareInformation.qwMemorySize') {
      $qw=[int64]$p.'HardwareInformation.qwMemorySize'
    }
  }
  if ($qw -gt 0) { $ram = $qw }
  $out += [pscustomobject]@{ name=$name; bytes=$ram }
}
$out | ConvertTo-Json -Compress
`
  const out = await powershell(script)
  // No output at all is PowerShell failing; empty output is Windows listing no adapters.
  if (out === null) return { gpus: [], ok: false }
  if (!out.trim()) return { gpus: [], ok: true }
  try {
    const parsed = JSON.parse(out.trim())
    const list = Array.isArray(parsed) ? parsed : [parsed]
    const gpus = list
      .filter((g: { name?: string; bytes?: number }) => g && g.name && !isVirtualAdapter(g.name))
      // An adapter reporting no memory cannot be budgeted against, so it is not a compute device.
      .filter((g: { bytes?: number }) => Number(g.bytes) > 0)
      .map((g: { name: string; bytes: number }, i: number) => ({
        index: i,
        name: g.name,
        vendor: guessVendor(g.name),
        totalVram: Number(g.bytes) || 0,
        // No measurement path on these adapters; the engine treats -1 as "estimate wide".
        freeVram: -1,
        utilisation: -1,
        freeIsMeasured: false
      }))
    return { gpus, ok: true }
  } catch {
    return { gpus: [], ok: false }
  }
}

/**
 * Screen-sharing and remote-desktop tools install display adapters that look like GPUs to WMI
 * but have no compute capability. Counting them produces phantom devices in the fit plan.
 */
const VIRTUAL_ADAPTER_PATTERNS = [
  /virtual/i,
  /parsec/i,
  /remote\s*display/i,
  /\bidd\b/i,
  /sunshine/i,
  /moonlight/i,
  /mirror\s*driver/i,
  /basic\s+display/i,
  /meta\b.*\bdriver/i
]

export function isVirtualAdapter(name: string): boolean {
  return VIRTUAL_ADAPTER_PATTERNS.some((re) => re.test(name))
}

function guessVendor(name: string): GpuDevice['vendor'] {
  const n = name.toLowerCase()
  if (n.includes('nvidia') || n.includes('geforce') || n.includes('rtx') || n.includes('quadro')) return 'nvidia'
  if (n.includes('amd') || n.includes('radeon')) return 'amd'
  if (n.includes('intel') || n.includes('arc')) return 'intel'
  return 'unknown'
}

export function pickBackend(gpus: GpuDevice[], cudaAvailable: boolean): Backend {
  if (cudaAvailable && gpus.some((g) => g.vendor === 'nvidia')) return 'cuda'
  if (gpus.some((g) => g.totalVram > 0)) return 'vulkan'
  return 'cpu'
}

let cachedCpuName: string | null = null

async function cpuName(): Promise<string> {
  if (cachedCpuName) return cachedCpuName
  const cpus = os.cpus()
  cachedCpuName = cpus.length ? cpus[0].model.trim() : 'unknown CPU'
  return cachedCpuName
}

/**
 * Merge NVIDIA-measured devices with the generic list so a mixed rig
 * (e.g. an RTX card plus an Intel iGPU) reports both, without duplicating the NVIDIA entries.
 */
function mergeDevices(nvidia: GpuDevice[], generic: GpuDevice[]): GpuDevice[] {
  if (!nvidia.length) return generic
  const others = generic.filter((g) => g.vendor !== 'nvidia')
  return [...nvidia, ...others.map((g, i) => ({ ...g, index: nvidia.length + i }))]
}

/**
 * Whether a detection is an answer or a guess.
 *
 * Provisional when a probe that should have spoken did not: nvidia-smi failing on a machine whose
 * display adapters include an NVIDIA card, or Windows not listing adapters at all when nothing else
 * answered either. A missing nvidia-smi on a machine with no NVIDIA card is a real answer, and so
 * is an empty list that every probe agrees on.
 */
export function verdictFor(
  nvidia: ProbeOutcome,
  generic: { ok: boolean; gpus: GpuDevice[] }
): NonNullable<HardwareSnapshot['detection']> {
  if (nvidia.status === 'failed' && (!generic.ok || generic.gpus.some((g) => g.vendor === 'nvidia'))) {
    return { state: 'provisional', reason: `nvidia-smi ${nvidia.reason}` }
  }
  if (!generic.ok && nvidia.status !== 'ok') {
    return { state: 'provisional', reason: 'Windows did not list the display adapters' }
  }
  return { state: 'measured' }
}

/**
 * Keep a measured snapshot rather than replace it with a guess.
 *
 * A later detection that comes back provisional would otherwise tell a machine already seen with
 * GPUs that it has none. The measured adapters stay, with the fresher memory figure from the new
 * probe; a provisional result only ever replaces another provisional one.
 */
export function mergeDetection(previous: HardwareSnapshot | null, next: HardwareSnapshot): HardwareSnapshot {
  if (next.detection?.state !== 'provisional' || !previous || previous.detection?.state === 'provisional') return next
  return { ...previous, freeRam: next.freeRam, takenAt: next.takenAt }
}

export async function detectHardware(): Promise<HardwareSnapshot> {
  const [nvidia, generic, name] = await Promise.all([detectNvidia(), detectGeneric(), cpuName()])

  const gpus = mergeDevices(nvidia.gpus, generic.gpus)
  // nvidia-smi answering with devices is what a usable CUDA runtime looks like; it was probed twice.
  const cuda = nvidia.outcome.status === 'ok' && nvidia.gpus.length > 0

  return {
    gpus,
    totalRam: os.totalmem(),
    freeRam: os.freemem(),
    cpuName: name,
    cpuThreads: os.cpus().length,
    backend: pickBackend(gpus, cuda),
    takenAt: Date.now(),
    detection: verdictFor(nvidia.outcome, generic)
  }
}

/**
 * Re-measure free VRAM immediately before a load.
 *
 * This is the P1 fix: a snapshot taken when the app started is stale by the time the user
 * clicks Load — they may have opened a game or a browser in between. We re-probe rather
 * than trusting the cached figure.
 */
export async function refreshFreeVram(snapshot: HardwareSnapshot): Promise<HardwareSnapshot> {
  const { gpus: nvidia } = await detectNvidia()
  if (!nvidia.length) return { ...snapshot, takenAt: Date.now() }

  const byName = new Map(nvidia.map((g) => [`${g.index}:${g.name}`, g]))
  const gpus = snapshot.gpus.map((g) => {
    const fresh = byName.get(`${g.index}:${g.name}`)
    return fresh ? { ...g, freeVram: fresh.freeVram, utilisation: fresh.utilisation } : g
  })
  return { ...snapshot, gpus, freeRam: os.freemem(), takenAt: Date.now() }
}
