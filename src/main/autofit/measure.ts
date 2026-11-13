/**
 * Exact memory figures for a plan, from llama.cpp itself.
 *
 * `llama-fit-params --fit-print on` builds the model and its context with nothing allocated and
 * reports what each device would hold: weights, cache, and compute buffers. It reads only the
 * file's layout, so it takes a second or two and loads no weights. These are the numbers the load
 * will allocate, which no formula in this app has ever matched across architectures: on
 * Qwen3.8-Flash-Next the compute buffers alone grow from 1.5 GB to 3.7 GB on one card between 64K
 * and 500K of context, through a sparse-attention indexer nothing here modelled.
 *
 * Results are kept in memory and on disk per model file, keyed by the exact arguments and the
 * server build that measured them, so a plan asked for again costs nothing.
 */

import { spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
import type { Backend, FitPlan, ModelRecord } from '@shared/types'
import { childEnv, llamaServerPath } from '../runtime/binaries'
import { APPDATA_DIR } from '../storage/paths'
import { logger } from '../log'
import { placementArgs } from './placement-args'

/** What llama.cpp would allocate, per device in its own order, and on the host. Bytes. */
export interface DryRun {
  devices: { name: string; model: number; context: number; compute: number }[]
  host: { model: number; context: number; compute: number }
}

const MiB = 1024 * 1024
const CACHE_DIR = path.join(APPDATA_DIR, 'fit-cache')
const memory = new Map<string, DryRun>()
/** Model files whose dry run failed, and until when not to try them again. A failure says nothing about other models. */
const failedUntil = new Map<string, number>()

function fitExe(backend: Backend): string {
  return path.join(path.dirname(llamaServerPath(backend)), 'llama-fit-params.exe')
}

function stamp(file: string): string {
  try {
    const st = fs.statSync(file)
    return `${st.size}:${Math.round(st.mtimeMs)}`
  } catch {
    return 'missing'
  }
}

function cacheFile(modelPath: string): string {
  return path.join(CACHE_DIR, `${crypto.createHash('sha256').update(modelPath).digest('hex').slice(0, 24)}.json`)
}

function readDisk(modelPath: string): Record<string, DryRun> {
  try {
    return JSON.parse(fs.readFileSync(cacheFile(modelPath), 'utf8')) as Record<string, DryRun>
  } catch {
    return {}
  }
}

function writeDisk(modelPath: string, key: string, result: DryRun): void {
  try {
    fs.mkdirSync(CACHE_DIR, { recursive: true })
    const all = readDisk(modelPath)
    all[key] = result
    // The most recent few hundred configurations of a model are plenty; drop the oldest beyond that.
    const keys = Object.keys(all)
    for (const k of keys.slice(0, Math.max(0, keys.length - 300))) delete all[k]
    fs.writeFileSync(cacheFile(modelPath), JSON.stringify(all))
  } catch (err) {
    logger.warn('autofit', 'could not keep a memory measurement', err)
  }
}

/** Parse `CUDA0 1356 2969 3736` lines (MiB) into a dry run. Null when nothing usable came back. */
export function parseFitPrint(stdout: string): DryRun | null {
  const devices: DryRun['devices'] = []
  let host: DryRun['host'] | null = null
  for (const line of stdout.split(/\r?\n/)) {
    const m = /^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$/.exec(line.trim())
    if (!m) continue
    const entry = { model: Number(m[2]) * MiB, context: Number(m[3]) * MiB, compute: Number(m[4]) * MiB }
    if (m[1] === 'Host') host = entry
    else devices.push({ name: m[1], ...entry })
  }
  return host ? { devices, host } : null
}

/** Whether exact measurement is available for this backend at all. */
export function canMeasure(backend: Backend, modelPath?: string): boolean {
  if (modelPath && Date.now() < (failedUntil.get(modelPath) ?? 0)) return false
  return backend !== 'cpu' && fs.existsSync(fitExe(backend))
}

/**
 * Measure a plan. Null when the tool is missing or fails, in which case the caller plans from
 * its own estimates as before.
 */
export async function measurePlan(model: ModelRecord, plan: FitPlan, backend: Backend): Promise<DryRun | null> {
  if (!canMeasure(backend, model.path)) return null
  const exe = fitExe(backend)
  const args = ['--fit-print', 'on', ...placementArgs(model.path, model.arch, plan)]
  const key = crypto
    .createHash('sha256')
    .update(JSON.stringify([args, stamp(model.path), stamp(exe), stamp(path.join(path.dirname(exe), 'llama.dll'))]))
    .digest('hex')

  const hit = memory.get(key) ?? readDisk(model.path)[key]
  if (hit) {
    memory.set(key, hit)
    return hit
  }

  const started = Date.now()
  const result = await new Promise<DryRun | null>((resolve) => {
    let out = ''
    let err = ''
    const child = spawn(exe, args, { windowsHide: true, env: childEnv() })
    const timer = setTimeout(() => child.kill(), 60_000)
    child.stdout.on('data', (d: Buffer) => (out += d.toString()))
    child.stderr.on('data', (d: Buffer) => {
      err += d.toString()
      if (err.length > 64 * 1024) err = err.slice(-32 * 1024)
    })
    child.on('error', () => {
      clearTimeout(timer)
      resolve(null)
    })
    child.on('close', (code) => {
      clearTimeout(timer)
      const parsed = code === 0 ? parseFitPrint(out) : null
      if (!parsed) logger.warn('autofit', `memory dry run failed (exit ${code})`, { tail: err.slice(-600) })
      resolve(parsed)
    })
  })

  if (!result) {
    // A model whose dry run fails will fail again; stop trying it for a while rather than costing every plan.
    failedUntil.set(model.path, Date.now() + 10 * 60_000)
    return null
  }
  logger.info('autofit', `measured a plan in ${Date.now() - started} ms`, {
    ctx: plan.contextLength,
    split: plan.layerSplit,
    cpuMoe: plan.cpuMoeLayers ?? 0,
    perDevice: result.devices.map((d) => Math.round((d.model + d.context + d.compute) / MiB))
  })
  memory.set(key, result)
  writeDisk(model.path, key, result)
  return result
}
