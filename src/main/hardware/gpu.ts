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

/** NVIDIA: exact free/total VRAM and utilisation. */
async function detectNvidia(): Promise<GpuDevice[]> {
  const out = await run('nvidia-smi', [
    '--query-gpu=index,name,memory.total,memory.free,utilization.gpu',
    '--format=csv,noheader,nounits'
  ])
  if (!out) return []

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

/**
 * Vendor-neutral fallback via WMI + the driver registry key.
 *
 * Win32_VideoController.AdapterRAM is a uint32 and therefore lies about anything over 4 GB,
 * so we prefer HardwareInformation.qwMemorySize from the display class registry key, which
 * is a true 64-bit value.
 */
async function detectGeneric(): Promise<GpuDevice[]> {
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
  if (!out) return []
  try {
    const parsed = JSON.parse(out.trim())
    const list = Array.isArray(parsed) ? parsed : [parsed]
    return list
      .filter((g: { name?: string; bytes?: number }) => g && g.name)
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
  } catch {
    return []
  }
}

function guessVendor(name: string): GpuDevice['vendor'] {
  const n = name.toLowerCase()
  if (n.includes('nvidia') || n.includes('geforce') || n.includes('rtx') || n.includes('quadro')) return 'nvidia'
  if (n.includes('amd') || n.includes('radeon')) return 'amd'
  if (n.includes('intel') || n.includes('arc')) return 'intel'
  return 'unknown'
}

/** Does a usable CUDA runtime exist? Presence of an NVIDIA adapter is not sufficient. */
async function hasCuda(): Promise<boolean> {
  const out = await run('nvidia-smi', ['--query-gpu=driver_version', '--format=csv,noheader'])
  return !!out && out.trim().length > 0
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

export async function detectHardware(): Promise<HardwareSnapshot> {
  const [nvidia, generic, cuda, name] = await Promise.all([
    detectNvidia(),
    detectGeneric(),
    hasCuda(),
    cpuName()
  ])

  const gpus = mergeDevices(nvidia, generic)

  return {
    gpus,
    totalRam: os.totalmem(),
    freeRam: os.freemem(),
    cpuName: name,
    cpuThreads: os.cpus().length,
    backend: pickBackend(gpus, cuda),
    takenAt: Date.now()
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
  const nvidia = await detectNvidia()
  if (!nvidia.length) return { ...snapshot, takenAt: Date.now() }

  const byName = new Map(nvidia.map((g) => [`${g.index}:${g.name}`, g]))
  const gpus = snapshot.gpus.map((g) => {
    const fresh = byName.get(`${g.index}:${g.name}`)
    return fresh ? { ...g, freeVram: fresh.freeVram, utilisation: fresh.utilisation } : g
  })
  return { ...snapshot, gpus, freeRam: os.freemem(), takenAt: Date.now() }
}
