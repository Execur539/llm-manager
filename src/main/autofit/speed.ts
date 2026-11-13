/**
 * How fast a plan will generate, in tokens per second.
 *
 * The old figure was a ratio out of 100 that nothing had ever been checked against. This one is an
 * estimate in real units, and once a model has run on this machine it is pulled towards what that
 * model actually did here: the server reports exact decode speed at the end of every reply, and the
 * app keeps those figures per plan.
 *
 * The estimate: each token reads the GPU-resident weights once, plus the routed experts it uses;
 * experts kept in system RAM are read from there unless the GPU expert cache already holds them.
 * Fitted to Qwen3.8-Flash-Next with every expert in RAM on a Ryzen 9900X with an RTX 5080 and
 * 4070 Ti: 18.1 tok/s without the cache, 37.4 with 128 slots (87% hits), 41.0 with 192 (92%).
 * That gives 22.4 ms a token of fixed and GPU cost and 32.8 ms to read every used expert from RAM.
 *
 * Pure, so it can be tested without a model.
 */

import type { FitPlan, ModelArchInfo } from '@shared/types'

/** One measured run: the plan's shape, and the decode speed the server reported. */
export interface SpeedSample {
  gpuLayers: number
  cpuMoeLayers: number
  expertCacheSlots: number
  draft: boolean
  contextLength: number
  tokensPerSecond: number
}

/** Effective system-RAM read rate for routed experts, from the fit above. */
const HOST_BYTES_PER_SECOND = 27e9
/** Effective VRAM read rate across both cards, including the cross-card hop. */
const GPU_BYTES_PER_SECOND = 400e9
/** Per-layer launch and synchronisation cost a token pays whatever the bytes. */
const PER_LAYER_SECONDS = 0.28e-3

/**
 * Share of used experts the GPU cache holds, for a cache of `slots` per layer out of `experts`.
 *
 * Routing is skewed: a few hundred most-recently-used experts of 512 cover nearly every token.
 * Through the two measured points (128 -> 87%, 192 -> 92%) with the obvious ends.
 */
export function cacheHitRate(slots: number, experts: number): number {
  if (slots <= 0 || experts <= 0) return 0
  if (slots >= experts - 1) return 1
  const x = slots / experts
  // 1 - (1 - x)^7: 86.7% at 128 of 512 and 96% at 192, against 87% and 92% measured.
  return Math.min(1, 1 - Math.pow(1 - x, 7))
}

/** Seconds a token takes under a plan, before any calibration. */
function secondsPerToken(arch: ModelArchInfo, plan: Pick<FitPlan, 'gpuLayers' | 'cpuMoeLayers' | 'expertCacheSlots'>): number {
  const n = arch.blockCount
  const dense = arch.layerDenseBytes ?? new Array<number>(n).fill(n > 0 ? arch.perLayerBytes / n : 0)
  const experts = arch.layerExpertBytes ?? new Array<number>(n).fill(0)
  const share = arch.expertCount > 0 && (arch.expertUsedCount ?? 0) > 0 ? (arch.expertUsedCount ?? 0) / arch.expertCount : 1
  const firstOnGpu = Math.max(0, n - plan.gpuLayers)
  const cpuMoe = plan.cpuMoeLayers ?? 0
  const hit = cacheHitRate(plan.expertCacheSlots ?? 0, arch.expertCount)
  let gpu = plan.gpuLayers > 0 ? (arch.outputBytes ?? 0) : 0
  let host = plan.gpuLayers > 0 ? 0 : (arch.outputBytes ?? 0)
  for (let il = 0; il < n; il++) {
    const used = experts[il] * share
    if (il < firstOnGpu) {
      // A whole block on the CPU reads its dense weights from RAM too, and computes there.
      host += dense[il] + used
      continue
    }
    gpu += dense[il]
    if (il < cpuMoe) {
      host += used * (1 - hit)
      gpu += used * hit
    } else {
      gpu += used
    }
  }
  return gpu / GPU_BYTES_PER_SECOND + host / HOST_BYTES_PER_SECOND + n * PER_LAYER_SECONDS
}

/** The estimate alone, in tokens per second. */
export function estimateSpeed(arch: ModelArchInfo, plan: Pick<FitPlan, 'gpuLayers' | 'cpuMoeLayers' | 'expertCacheSlots'>): number {
  const s = secondsPerToken(arch, plan)
  return s > 0 ? 1 / s : 0
}

/** Whether a measured run is the same plan, near enough that its figure can be quoted as is. */
function sameShape(a: SpeedSample, plan: Pick<FitPlan, 'gpuLayers' | 'cpuMoeLayers' | 'expertCacheSlots' | 'draftMax'>): boolean {
  const slots = plan.expertCacheSlots ?? 0
  return (
    a.gpuLayers === plan.gpuLayers &&
    a.cpuMoeLayers === (plan.cpuMoeLayers ?? 0) &&
    a.draft === (plan.draftMax ?? 0) > 0 &&
    Math.abs(a.expertCacheSlots - slots) <= Math.max(4, slots * 0.1)
  )
}

function median(values: number[]): number {
  const v = [...values].sort((a, b) => a - b)
  return v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2
}

/**
 * Expected tokens per second for a plan, and where the figure comes from.
 *
 *   measured    runs of this very plan shape on this machine: their median
 *   calibrated  runs of other shapes of this model: the estimate scaled by how far off it was for them
 *   estimate    no runs of this model yet
 */
export function predictSpeed(
  arch: ModelArchInfo,
  plan: Pick<FitPlan, 'gpuLayers' | 'cpuMoeLayers' | 'expertCacheSlots' | 'draftMax'>,
  samples: readonly SpeedSample[]
): { tokensPerSecond: number; source: 'measured' | 'calibrated' | 'estimate' } {
  const same = samples.filter((s) => sameShape(s, plan))
  if (same.length >= 2) return { tokensPerSecond: median(same.map((s) => s.tokensPerSecond)), source: 'measured' }
  const base = estimateSpeed(arch, plan)
  // Drafting changes speed by content, so it is calibrated only against runs that drafted the same way.
  const comparable = samples.filter((s) => s.draft === (plan.draftMax ?? 0) > 0)
  const pool = comparable.length ? comparable : samples
  if (!pool.length) return { tokensPerSecond: base, source: 'estimate' }
  const ratios = pool
    .map((s) => s.tokensPerSecond / estimateSpeed(arch, s))
    .filter((r) => Number.isFinite(r) && r > 0)
  if (!ratios.length) return { tokensPerSecond: base, source: 'estimate' }
  return { tokensPerSecond: base * median(ratios), source: 'calibrated' }
}
