/**
 * Planning a load from measurements rather than formulas.
 *
 * The engine still searches the space of placements, because it is fast and it can try thousands
 * of them. What changed is what it searches against:
 *
 *   llama.cpp's buffers   The engine's estimate is checked with llama.cpp's own dry run (measure.ts),
 *                         and the difference is fed back as a correction until the chosen plan is
 *                         one llama.cpp has confirmed. Three or four dry runs of 1.5 s each.
 *   the expert cache      Counted exactly from the model file: (slots + 1) experts per layer.
 *   everything else       The CUDA runtime, the vision projector and a separate MTP head, measured
 *                         from the model's previous load (what the cards lost, less the two above).
 *   what is kept free     The user's reserve, plus a small stated allowance. Nothing hidden.
 *
 * Speed comes from the same idea: an estimate in tokens per second, pulled towards what this model
 * has actually done on this machine (speed.ts).
 */

import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
import type { FitConstraints, FitPlan, FitResult, GpuDevice, HardwareSnapshot, ModelRecord, PlanCorrection } from '@shared/types'
import { APPDATA_DIR } from '../storage/paths'
import { all, get, run } from '../storage/db'
import { logger } from '../log'
import { DEFAULT_SAFETY, correctionFor, fmtBytes, gpuKey, planFit, usableDevices } from './engine'
import { canMeasure, measurePlan, type DryRun } from './measure'
import { predictSpeed, type SpeedSample } from './speed'

const MB = 1024 * 1024
/** The allowance once both llama.cpp's buffers and the extras have been measured for a plan. */
export const MEASURED_SAFETY = 160 * MB
/** Dry runs per plan request, at most: a request that has not converged by then plans from the estimate. */
const MAX_DRY_RUNS = 6

// ---------------------------------------------------------------- corrections, kept per model file

function correctionsFile(model: ModelRecord): string {
  const id = crypto.createHash('sha256').update(model.path).digest('hex').slice(0, 24)
  return path.join(APPDATA_DIR, 'fit-cache', `${id}-corrections.json`)
}

function fileStamp(file: string): string {
  try {
    const st = fs.statSync(file)
    return `${st.size}:${Math.round(st.mtimeMs)}`
  } catch {
    return 'missing'
  }
}

function loadCorrections(model: ModelRecord): PlanCorrection[] {
  try {
    const raw = JSON.parse(fs.readFileSync(correctionsFile(model), 'utf8')) as { stamp?: string; entries?: PlanCorrection[] }
    // Measured against a different file is measured against a different model.
    return raw.stamp === fileStamp(model.path) && Array.isArray(raw.entries) ? raw.entries : []
  } catch {
    return []
  }
}

function saveCorrections(model: ModelRecord, entries: PlanCorrection[]): void {
  try {
    const file = correctionsFile(model)
    fs.mkdirSync(path.dirname(file), { recursive: true })
    fs.writeFileSync(file, JSON.stringify({ stamp: fileStamp(model.path), entries: entries.slice(-80) }))
  } catch (err) {
    logger.warn('autofit', 'could not keep plan corrections', err)
  }
}

/** One configuration, as llama.cpp sees it. Plans with the same key allocate the same buffers. */
export function configKey(p: Pick<FitPlan, 'contextLength' | 'kvType' | 'kvTypeV' | 'batchSize' | 'ubatchSize' | 'flashAttention' | 'gpuLayers' | 'cpuMoeLayers' | 'layerSplit'>): string {
  return [
    p.contextLength,
    p.kvType,
    p.kvTypeV ?? p.kvType,
    p.ubatchSize ?? p.batchSize,
    p.flashAttention ? 1 : 0,
    p.gpuLayers,
    p.cpuMoeLayers ?? 0,
    (p.layerSplit ?? []).join('+')
  ].join('|')
}

function correctionKey(c: PlanCorrection): string {
  return [c.contextLength, c.kvType, c.kvTypeV, c.ubatch, c.flashAttention ? 1 : 0, c.gpuLayers, c.cpuMoeLayers, c.layerSplit.join('+')].join('|')
}

/** Same configuration and context, any split: a newer measurement of it replaces the older. */
function slotKey(c: PlanCorrection): string {
  return [c.contextLength, c.kvType, c.kvTypeV, c.ubatch, c.flashAttention ? 1 : 0, c.gpuLayers, c.cpuMoeLayers].join('|')
}

/**
 * The correction one dry run gives: what llama.cpp said, less what the bare estimate said.
 *
 * The plan's figures already include whatever correction was applied when it was made, so that is
 * taken back out first; otherwise each measurement would correct the last correction.
 */
export function correctionFromDryRun(plan: FitPlan, dry: DryRun, applied: readonly PlanCorrection[], cards: number): PlanCorrection | null {
  const llama = plan.llamaVramPerGpu
  if (!llama || dry.devices.length !== cards || llama.length !== cards) return null
  const split = plan.layerSplit ?? []
  const ubatch = plan.ubatchSize ?? plan.batchSize
  const used = correctionFor(
    applied,
    {
      contextLength: plan.contextLength,
      kvType: { k: plan.kvType, v: plan.kvTypeV ?? plan.kvType },
      ubatch,
      flashAttention: plan.flashAttention,
      gpuLayers: plan.gpuLayers,
      cpuMoeLayers: plan.cpuMoeLayers ?? 0
    },
    cards
  )
  const offset = dry.devices.map((d, i) => {
    const holds = (split[i] ?? 1) > 0
    const estimate = llama[i] - (holds ? used[i] : 0)
    return d.model + d.context + d.compute - estimate
  })
  return {
    contextLength: plan.contextLength,
    kvType: plan.kvType,
    kvTypeV: plan.kvTypeV ?? plan.kvType,
    ubatch,
    flashAttention: plan.flashAttention,
    gpuLayers: plan.gpuLayers,
    cpuMoeLayers: plan.cpuMoeLayers ?? 0,
    layerSplit: split,
    offset
  }
}

// ---------------------------------------------------------------- extras, learned per model

/** What a load's extras depend on: change any of these and the last measurement no longer applies. */
export interface ExtrasFeatures {
  mmproj: string | null
  draft: string | null
  ubatch: number
}

export interface StoredExtras {
  keys: string[]
  bytes: number[]
  features: ExtrasFeatures
  contextLength: number
  /** False when llama.cpp's part was estimated rather than measured, so the extras absorbed its error. */
  exact: boolean
}

function sameFeatures(a: ExtrasFeatures, b: ExtrasFeatures): boolean {
  return a.mmproj === b.mmproj && a.draft === b.draft && a.ubatch === b.ubatch
}

/**
 * The extras the model's last load measured, per card in the plan's order, or undefined.
 *
 * A separate MTP head keeps a layer of cache for the whole context, so a load at a different context
 * moves its figure on the first card; `draftKvAt` says by how much.
 */
export function learnedExtras(
  modelId: string,
  cards: GpuDevice[],
  features: ExtrasFeatures,
  contextLength: number,
  draftKvAt: (ctx: number) => number
): number[] | undefined {
  const row = get<{ extras: string | null }>('SELECT extras FROM model_configs WHERE model_id = ?', modelId)
  if (!row?.extras) return undefined
  try {
    const stored = JSON.parse(row.extras) as StoredExtras
    if (!stored.features || !sameFeatures(stored.features, features)) return undefined
    const out = cards.map((g) => {
      const i = stored.keys.indexOf(gpuKey(g))
      return i >= 0 ? stored.bytes[i] : NaN
    })
    if (out.some((v) => !Number.isFinite(v))) return undefined
    if (features.draft && out.length) out[0] = Math.max(0, out[0] + draftKvAt(contextLength) - draftKvAt(stored.contextLength))
    return out
  } catch {
    return undefined
  }
}

/**
 * The extras a load has just revealed: what each card lost, less llama.cpp's buffers and the cache.
 */
export function extrasFromLoad(
  plan: FitPlan,
  cards: GpuDevice[],
  actual: number[],
  features: ExtrasFeatures
): StoredExtras | null {
  const llama = plan.llamaVramPerGpu
  const cache = plan.cacheVramPerGpu
  if (!llama || !cache || actual.some((v) => !(v > 0))) return null
  return {
    keys: cards.map(gpuKey),
    bytes: actual.map((a, d) => Math.max(0, Math.round(a - llama[d] - cache[d]))),
    features,
    contextLength: plan.contextLength,
    exact: !!plan.measured
  }
}

// ---------------------------------------------------------------- speed, learned per model

export function speedSamples(modelId: string): SpeedSample[] {
  return all<{
    gpu_layers: number
    cpu_moe_layers: number
    cache_slots: number
    draft: number
    context_length: number
    tokens_per_second: number
  }>(
    'SELECT gpu_layers, cpu_moe_layers, cache_slots, draft, context_length, tokens_per_second FROM plan_speeds WHERE model_id = ? ORDER BY created_at DESC LIMIT 200',
    modelId
  ).map((r) => ({
    gpuLayers: r.gpu_layers,
    cpuMoeLayers: r.cpu_moe_layers,
    expertCacheSlots: r.cache_slots,
    draft: !!r.draft,
    contextLength: r.context_length,
    tokensPerSecond: r.tokens_per_second
  }))
}

/** Keep one reply's decode speed against the plan that produced it. */
export function recordSpeed(modelId: string, plan: FitPlan, tokensPerSecond: number, tokens: number): void {
  if (!(tokensPerSecond > 0) || tokens < 32) return
  run(
    'INSERT INTO plan_speeds (model_id, gpu_layers, cpu_moe_layers, cache_slots, draft, context_length, tokens_per_second, tokens, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
    modelId,
    plan.gpuLayers,
    plan.cpuMoeLayers ?? 0,
    plan.expertCacheSlots ?? 0,
    (plan.draftMax ?? 0) > 0 ? 1 : 0,
    plan.contextLength,
    tokensPerSecond,
    tokens,
    Date.now()
  )
  // A few hundred replies per model is plenty to judge by.
  run(
    'DELETE FROM plan_speeds WHERE model_id = ? AND created_at < (SELECT MIN(created_at) FROM (SELECT created_at FROM plan_speeds WHERE model_id = ? ORDER BY created_at DESC LIMIT 300))',
    modelId,
    modelId
  )
}

// ---------------------------------------------------------------- the plan itself

const isPlan = (p: FitPlan | null | undefined): p is FitPlan => !!p

/**
 * Plan a model's load.
 *
 * `measure` asks for the confirmed plan, which costs a few seconds of dry runs the first time a
 * configuration is seen and nothing after. Without it the plan is the estimate, corrected by
 * whatever has already been measured for this model, which is what the library's badges use.
 */
export async function planModel(req: {
  model: ModelRecord
  hw: HardwareSnapshot
  constraints: FitConstraints
  measure: boolean
}): Promise<FitResult> {
  const arch = req.model.arch
  if (!arch) throw new Error(req.model.error ?? 'Model metadata could not be parsed')
  const cards = usableDevices(req.hw)
  let corrections = loadCorrections(req.model)
  const measuring = req.measure && canMeasure(req.hw.backend, req.model.path)
  const constraints: FitConstraints = {
    ...req.constraints,
    safetyBytes:
      req.constraints.safetyBytes ?? (measuring && req.constraints.extrasPerGpu ? MEASURED_SAFETY : DEFAULT_SAFETY)
  }

  let result = planFit(arch, req.hw, { ...constraints, corrections })

  if (measuring) {
    let runs = 0
    let changed = false
    while (runs < MAX_DRY_RUNS) {
      const known = new Set(corrections.map(correctionKey))
      // The chosen plan first: it is the one that will load. Alternatives are confirmed too, so the
      // figures beside every button are llama.cpp's, as long as the budget of dry runs allows.
      const targets = [result.chosen, ...result.alternatives]
        .filter(isPlan)
        .filter((p) => p.gpuLayers > 0 && !known.has(configKey(p)))
      if (!targets.length) break
      const applied = corrections
      let added = 0
      for (const plan of targets) {
        if (runs >= MAX_DRY_RUNS) break
        runs++
        const dry = await measurePlan(req.model, plan, req.hw.backend)
        if (!dry) break
        const c = correctionFromDryRun(plan, dry, applied, cards.length)
        if (!c) continue
        corrections = [...corrections.filter((x) => slotKey(x) !== slotKey(c)), c]
        added++
      }
      if (!added) break
      changed = true
      result = planFit(arch, req.hw, { ...constraints, corrections })
    }
    if (changed) saveCorrections(req.model, corrections)
  }

  const known = new Set(corrections.map(correctionKey))
  const samples = speedSamples(req.model.id)
  const annotate = (p: FitPlan): FitPlan => {
    const speed = predictSpeed(arch, p, samples)
    return {
      ...p,
      measured: known.has(configKey(p)),
      tokensPerSecond: Math.round(speed.tokensPerSecond * 10) / 10,
      speedSource: speed.source
    }
  }
  const notes = [...result.notes]
  if (measuring && result.chosen && !known.has(configKey(result.chosen))) {
    notes.push('llama.cpp could not confirm this plan, so its memory figures are the estimate.')
  }
  return {
    ...result,
    chosen: result.chosen ? annotate(result.chosen) : null,
    alternatives: result.alternatives.map(annotate),
    notes
  }
}

/** A sentence on how a load compared with its plan, or null when it was close. */
export function describeLoad(plan: FitPlan, cards: GpuDevice[], actual: number[]): string | null {
  // A reading of zero means the measurement failed (or a stand-in server), not that nothing loaded.
  if (!actual.length || actual.some((v) => !(v > 0))) return null
  const predicted = plan.predictedVramPerGpu
  const worst = Math.max(...actual.map((a, d) => Math.abs(a - (predicted[d] ?? 0))))
  if (!Number.isFinite(worst) || worst < 300 * MB) return null
  const pairs = cards.map((g, d) => `${g.name.replace(/^NVIDIA GeForce /, '')} ${fmtBytes(actual[d] ?? 0)} (planned ${fmtBytes(predicted[d] ?? 0)})`)
  return `This load used ${pairs.join(', ')}. The difference is now measured, so the next plan for this model uses it.`
}
