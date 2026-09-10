/**
 * The auto-fit engine.
 *
 * Objective (from the plan): maximise context length, subject to KV quant >= Q4 (prefer Q8),
 * targeting >=64K and ideally 128K where the model allows — without ever OOMing and without
 * ever silently degrading behind the user's back.
 *
 * The four failure modes this exists to beat:
 *   P1  sizing against total VRAM instead of free VRAM
 *   P2  naive even splits across mismatched GPUs
 *   P3  over-conservative offload that leaves VRAM unused
 *   P4  ignoring KV growth, so a load succeeds then OOMs mid-chat
 *
 * P4 is handled structurally: we reserve the KV cache for the *entire* configured context
 * up front, so a load that succeeds cannot die as the conversation grows.
 */

import type {
  Backend,
  FitConstraints,
  FitPlan,
  FitResult,
  GpuDevice,
  HardwareSnapshot,
  KvType,
  ModelArchInfo
} from '@shared/types'

const MB = 1024 * 1024
const GB = 1024 * MB

/** Bytes per KV element for each cache type. Derived from ggml block layouts. */
const KV_ELEMENT_BYTES: Record<KvType, number> = {
  f16: 2,
  q8_0: 34 / 32, // 1.0625
  q4_0: 18 / 32 // 0.5625
}

const KV_ORDER: KvType[] = ['f16', 'q8_0', 'q4_0']

/**
 * How far past its trained length a model may be stretched, when stretching is allowed at all.
 *
 * Four is what the YaRN paper and the Qwen documentation both work in, and it is where the
 * published evidence stops rather than where the arithmetic does.
 */
const MAX_ROPE_SCALE = 4

/**
 * Where a KV type sits in the quality order, with a fallback for one that is not in it at all.
 *
 * `indexOf` returns -1 for an unrecognised value, and -1 fed to `slice` counts from the end —
 * which silently produces a plausible-looking but unrelated candidate list rather than failing.
 */
function clampToOrder(kv: KvType, fallback: number): number {
  const i = KV_ORDER.indexOf(kv)
  return i === -1 ? fallback : i
}

/** A cache precision for the keys and one for the values, which need not be the same. */
export interface KvChoice {
  k: KvType
  v: KvType
}

export function kvLabel(kv: KvChoice): string {
  return kv.k === kv.v ? kv.k : `${kv.k}/${kv.v}`
}

/**
 * Every key/value pairing worth trying, largest first.
 *
 * The keys are held at or above the floor and the values are free to go lower, which is the whole
 * point: they are not equally sensitive. Values quantised to four bits change roughly one answer
 * in five hundred; keys quantised to four bits have been measured taking a model from 92% to
 * 24.2%. A single lever for both could only offer the cliff.
 *
 * Ordered by bytes per element rather than by nesting, so stepping down the list is monotonic in
 * memory and the engine's "step down to buy context" logic reads the same as before.
 */
function kvLadder(preferred: KvType, floor: KvType): KvChoice[] {
  const floorIndex = clampToOrder(floor, KV_ORDER.length - 1)
  const preferredIndex = Math.min(clampToOrder(preferred, 0), floorIndex)
  const pairs: KvChoice[] = []
  for (const k of KV_ORDER.slice(preferredIndex, floorIndex + 1)) {
    // Values never finer than the keys: spending more on the tolerant half would be backwards.
    for (const v of KV_ORDER.slice(KV_ORDER.indexOf(k))) pairs.push({ k, v })
  }
  const size = (c: KvChoice): number => KV_ELEMENT_BYTES[c.k] + KV_ELEMENT_BYTES[c.v]
  return pairs.sort((a, b) => size(b) - size(a))
}

/** CUDA runtime + cuBLAS workspace claimed per device before any weights load. */
const CUDA_CONTEXT_OVERHEAD = 350 * MB
/** Graph/scratch allocations that don't scale with batch or context. */
const BASE_GRAPH_OVERHEAD = 96 * MB
/**
 * When free VRAM cannot be measured (AMD/Intel on Windows), assume this fraction of total
 * is actually available. Deliberately conservative: an OOM is worse than a smaller context.
 */
const UNMEASURED_FREE_FRACTION = 0.85
/**
 * Correction applied to the computed recurrent-state size.
 *
 * The naive conv+state figure under-counts what llama.cpp actually reserves for the `rs cache`
 * on hybrid models — measured against Qwen3.8-27B, which OOMed on a plan that looked like it
 * had 1.4 GB to spare. Erring high here costs a little context; erring low costs a failed load.
 */
const SSM_STATE_SAFETY = 2.5
/**
 * The projector's runtime footprint exceeds its file size: it allocates image-tile buffers
 * on top of its weights. Measured against Qwen3.8-27B's 0.59 GB Q8_0 projector.
 */
/*
 * What to reserve for a vision projector, as a multiple of its file size.
 *
 * The weights are only part of it: the encoder also needs a compute buffer, and that is what
 * actually ran out. Raised from 1.8 after a load where 1.8x left device 0 unable to find the
 * 248 MiB the projector asked for — llama.cpp logged the failure, reported the model loaded
 * anyway, and then segfaulted on the first video. An over-reservation costs some context; an
 * under-reservation costs the model.
 */
const MMPROJ_OVERHEAD = 2.4

/** Floor for the projector's compute buffer, independent of how small the projector itself is. */
const MMPROJ_MIN_COMPUTE = 512 * 1024 * 1024

export const DEFAULT_CONSTRAINTS: FitConstraints = {
  /*
   * The floor applies to the keys, which are the half that cannot be pushed far.
   *
   * Left at q4_0 so nothing that used to fit stops fitting, but it is now the last rung rather
   * than the second. Stepping down used to go straight from q8_0 on both halves to q4_0 on both,
   * and four-bit keys are the one setting measured to break a model outright. The ladder now has
   * q8_0 keys with q4_0 values in between, which is where almost all of the saving was anyway --
   * so q4_0 keys are only reached when nothing else fits at all, and the plan says so.
   */
  minKvType: 'q4_0',
  /*
   * q8_0 rather than f16, still.
   *
   * Its measured cost is a rounding error -- on the order of 0.002 to 0.05 perplexity, and zero
   * changed answers across sixteen hundred deep-context comparisons -- and it halves the cache.
   * Preferring f16 would make the first plan that clears the target the one with the least
   * context, which is the opposite of the point.
   */
  preferredKvType: 'q8_0',
  targetContext: 65536,
  /*
   * As much as the model was trained for.
   *
   * This was a fixed 131,072 and it was the real ceiling on every load: a model trained to
   * 262,144 was planned at half its context regardless of how much VRAM was free, because the
   * ideal never asked for more. The trained length is the honest ceiling -- past it the model is
   * extrapolating, which is a decision to be taken deliberately rather than by a default.
   *
   * A large finite number rather than Infinity: this value reaches a binary search as its upper
   * bound, and an infinite bound never converges.
   */
  idealContext: 1_048_576,
  headroomBytes: 768 * MB,
  overrides: {}
}

/**
 * KV cache bytes for a given context length.
 *
 * 2 (K and V) x attention-layers x kv_heads x head_dim x tokens x bytes-per-element.
 *
 * Two things make this smaller than a naive estimate:
 *   - GQA models have far fewer KV heads than attention heads, which is why a modern 27B can
 *     hold a 128K context that an older MHA 13B cannot.
 *   - Hybrid models only cache on their *attention* layers. Qwen3.8-27B has 64 blocks but a
 *     `full_attention_interval` of 4, so just 16 of them grow with context. Counting all 64
 *     overestimates the cache roughly fourfold and needlessly shrinks the planned context.
 *
 * The recurrent state on SSM layers is added separately: it is real memory, but a fixed amount
 * that does not scale with context.
 */
export function kvCacheBytes(arch: ModelArchInfo, contextLength: number, kv: KvType | KvChoice): number {
  // Older metadata (or a model parsed before hybrid support) leaves attentionLayers unset.
  const attentionLayers = arch.attentionLayers || arch.blockCount
  /*
   * K and V are sized separately, because they do not tolerate quantisation equally.
   *
   * Measured across llama.cpp's own comparisons, q4_0 on the values changes about one answer in
   * five hundred, while q4_0 on the keys can collapse a model outright — one Qwen 2.5 test went
   * from 92% to 24.2% accuracy. Charging both at the same rate hid that asymmetry and made
   * "step the cache down" a single lever with a cliff hidden in the middle of it.
   */
  const choice = typeof kv === 'string' ? { k: kv, v: kv } : kv
  const perTokenPerLayer =
    (arch.headCountKv * arch.headDim * KV_ELEMENT_BYTES[choice.k]) +
    (arch.headCountKv * arch.headDim * KV_ELEMENT_BYTES[choice.v])
  const attentionCache = perTokenPerLayer * attentionLayers * contextLength
  // The recurrent state is allocated per sequence slot. We run llama-server with --parallel 1,
  // so this is one slot's worth — but it is measured empirically to be larger than the naive
  // conv+state figure, so a correction factor keeps the estimate on the safe side.
  const recurrentState = (arch.ssmLayers ?? 0) * (arch.ssmStateBytesPerLayer ?? 0) * SSM_STATE_SAFETY
  return attentionCache + recurrentState
}

/**
 * Compute-buffer estimate.
 *
 * The dominant term without flash attention is the attention score matrix, which is
 * O(batch x context x heads) and explodes at the 64K-128K contexts we target — at 128K it
 * alone can exceed 4 GB. This is why the engine turns flash attention on by default and
 * says so in the rationale rather than silently.
 */
export function computeBufferBytes(
  arch: ModelArchInfo,
  contextLength: number,
  batchSize: number,
  flashAttention: boolean
): number {
  const activations = batchSize * arch.embeddingLength * 2 * 6
  const attention = flashAttention ? batchSize * arch.headDim * arch.headCount * 2 * 4 : batchSize * contextLength * arch.headCount * 2
  return activations + attention + BASE_GRAPH_OVERHEAD
}

/**
 * The logits buffer, which lives only on the device holding the output layer.
 *
 * llama.cpp reserves its worst-case graph with an output row for every token of the physical
 * batch, so this is batch x vocabulary x 4 bytes — half a gigabyte for a 248K vocabulary. It was
 * sized for a single token and charged to every device, which under-counted the card that
 * actually holds it; that was one of the two errors behind the 1.2.0 plan that would not load.
 */
export function logitsBufferBytes(arch: ModelArchInfo, batchSize: number): number {
  // The physical batch is llama.cpp's default 512 whatever the logical batch; this app never sets it.
  return Math.min(batchSize, 512) * arch.vocabSize * 4
}

/**
 * Can this device actually participate under the selected backend?
 *
 * This matters on mixed rigs. Under CUDA, only NVIDIA devices are addressable — handing
 * llama.cpp a tensor split that includes an AMD iGPU produces a broken invocation, and even
 * on Vulkan an integrated GPU sharing system RAM is slower than spilling to the host, so it
 * should never be given a share while discrete cards are present.
 */
export function usableForBackend(gpu: GpuDevice, backend: Backend, hasDiscrete: boolean): boolean {
  if (gpu.totalVram <= 0) return false
  if (backend === 'cuda') return gpu.vendor === 'nvidia'
  if (backend === 'cpu') return false
  // Vulkan: an iGPU is only worth using when it is the only thing available.
  if (hasDiscrete && isIntegrated(gpu)) return false
  return true
}

/** Integrated GPUs share system RAM; their "VRAM" is not a separate pool. */
function isIntegrated(gpu: GpuDevice): boolean {
  return /\bgraphics\b/i.test(gpu.name) && !/\b(rtx|gtx|rx|arc)\b/i.test(gpu.name)
}

/** Usable bytes on a device after headroom and fixed runtime overhead. */
function deviceBudget(gpu: GpuDevice, headroomBytes: number, backend: string): number {
  // P1: prefer the measured free figure; fall back to a discounted total only when we
  // genuinely have no way to measure, and never to the raw total.
  const free = gpu.freeIsMeasured && gpu.freeVram >= 0 ? gpu.freeVram : gpu.totalVram * UNMEASURED_FREE_FRACTION
  const overhead = backend === 'cuda' ? CUDA_CONTEXT_OVERHEAD : 0
  return Math.max(0, free - headroomBytes - overhead)
}

/**
 * P2: split proportional to *actual free capacity*, not device count.
 * A 24 GB + 8 GB pair gets 0.75/0.25, not 0.5/0.5.
 */
export function proportionalSplit(budgets: number[]): number[] {
  const total = budgets.reduce((a, b) => a + b, 0)
  if (total <= 0) return budgets.map(() => 0)
  return budgets.map((b) => b / total)
}

interface Attempt {
  fits: boolean
  perGpu: number[]
  /** Blocks per GPU, with the output layer counted on the card that takes the last block. */
  layerSplit: number[]
  hostBytes: number
  kvBytes: number
  weightsOnGpu: number
  overhead: number
}

/** What each block weighs, and what always sits outside the blocks. */
interface LayerCosts {
  dense: number[]
  experts: number[]
  input: number
  output: number
}

function layerCosts(arch: ModelArchInfo): LayerCosts {
  const n = arch.blockCount
  const dense = arch.layerDenseBytes
  const experts = arch.layerExpertBytes
  if (dense && experts && dense.length === n && experts.length === n) {
    return {
      dense,
      experts,
      input: arch.inputBytes ?? 0,
      output: arch.outputBytes ?? Math.max(0, arch.nonLayerBytes - (arch.inputBytes ?? 0))
    }
  }
  /*
   * A record parsed before the per-block breakdown existed.
   *
   * Spread the block bytes evenly and charge everything outside the blocks to the output layer,
   * which is what this engine always assumed — so an old record plans the way it always did until
   * the library rescans it.
   */
  const per = n > 0 ? arch.perLayerBytes / n : 0
  return {
    dense: new Array<number>(n).fill(per),
    experts: new Array<number>(n).fill(0),
    input: 0,
    output: arch.nonLayerBytes
  }
}

/** Blocks that carry routed experts, and one past the last of them: the most `--n-cpu-moe` can usefully be. */
function expertSpan(costs: LayerCosts): { blocks: number; end: number } {
  let blocks = 0
  let end = 0
  costs.experts.forEach((bytes, il) => {
    if (bytes > 0) {
      blocks++
      end = il + 1
    }
  })
  return { blocks, end }
}

/**
 * Cut the offloaded items — blocks in order, then the output layer — into one run per GPU.
 *
 * In proportion to each card's free capacity (P2), but by what each item weighs rather than by
 * how many there are. Counting blocks assumed they all weigh the same, which an MoE model with
 * some experts in system RAM does not, and a vocabulary head several times a block's size breaks
 * on whichever card receives it.
 */
function cutProportional(items: number[], caps: number[]): number[] {
  const counts = new Array<number>(caps.length).fill(0)
  const capTotal = caps.reduce((a, c) => a + Math.max(0, c), 0)
  const total = items.reduce((a, b) => a + b, 0)
  const ends: number[] = []
  let acc = 0
  for (const c of caps) {
    acc += capTotal > 0 ? (Math.max(0, c) / capTotal) * total : total / caps.length
    ends.push(acc)
  }
  let run = 0
  let dev = 0
  for (const item of items) {
    // An item goes to the card its midpoint falls on, so a boundary splits the difference.
    while (dev < caps.length - 1 && run + item / 2 > ends[dev]) dev++
    counts[dev]++
    run += item
  }
  return counts
}

/** Fill each card before starting the next: the fallback when the even cut overflows one of them. */
function cutGreedy(items: number[], caps: number[]): number[] {
  const counts = new Array<number>(caps.length).fill(0)
  let dev = 0
  let used = 0
  for (const item of items) {
    while (dev < caps.length - 1 && used + item > caps[dev]) {
      dev++
      used = 0
    }
    counts[dev]++
    used += item
  }
  return counts
}

/**
 * Put the offloaded items on the cards and work out what each card then needs.
 *
 * Every card holding at least one item also holds a compute buffer, and the card that ends up
 * with the output layer holds the logits buffer too. A placement that fits matters more than an
 * even one, so a front-loaded fill is used when the even cut does not fit.
 */
function placeItems(
  items: number[],
  budgets: number[],
  compute: number,
  logits: number
): { counts: number[]; perGpu: number[] } {
  const devices = budgets.length
  const caps = budgets.map((b, d) => b - compute - (d === devices - 1 ? logits : 0))
  const measure = (counts: number[]): number[] => {
    const need = new Array<number>(devices).fill(0)
    let k = 0
    let outputDevice = 0
    for (let d = 0; d < devices; d++) {
      for (let j = 0; j < counts[d]; j++) need[d] += items[k++]
      if (counts[d] > 0) {
        need[d] += compute
        outputDevice = d
      }
    }
    need[outputDevice] += logits
    return need
  }
  const fitsAll = (need: number[]): boolean => need.every((x, d) => x <= budgets[d])

  const even = cutProportional(items, caps)
  const evenNeed = measure(even)
  if (devices === 1 || fitsAll(evenNeed)) return { counts: even, perGpu: evenNeed }

  const filled = cutGreedy(items, caps)
  const filledNeed = measure(filled)
  return fitsAll(filledNeed) ? { counts: filled, perGpu: filledNeed } : { counts: even, perGpu: evenNeed }
}

/**
 * Try a placement at a given context length.
 *
 * Mirrors what llama.cpp does rather than an idealised split:
 *   - the input layer (token embedding, per-layer embedding tables) always stays in system RAM;
 *   - blocks are offloaded from the *end* of the stack, so `gpuLayers` are the last ones;
 *   - the output layer follows them onto the last card whenever any block is offloaded;
 *   - `--n-cpu-moe N` keeps the routed experts of the *first* N blocks in system RAM, wherever the
 *     rest of those blocks are;
 *   - each block's slice of the KV cache lives with the block.
 */
function attempt(
  arch: ModelArchInfo,
  hw: HardwareSnapshot,
  budgets: number[],
  contextLength: number,
  kvType: KvChoice,
  gpuLayers: number,
  batchSize: number,
  flashAttention: boolean,
  cpuMoeLayers = 0
): Attempt {
  const totalLayers = arch.blockCount
  const costs = layerCosts(arch)
  const kvTotal = kvCacheBytes(arch, contextLength, kvType)
  // Attention layers are interleaved through the stack, so an even spread is how the cache lands.
  const kvPerBlock = totalLayers > 0 ? kvTotal / totalLayers : 0
  const firstOnGpu = Math.max(0, totalLayers - gpuLayers)
  const cpuMoe = Math.max(0, Math.min(cpuMoeLayers, totalLayers))

  let hostBytes = costs.input
  let weightsOnGpu = 0
  const items: number[] = []
  for (let il = 0; il < totalLayers; il++) {
    const expertsOnHost = il < cpuMoe
    if (il < firstOnGpu) {
      hostBytes += costs.dense[il] + costs.experts[il] + kvPerBlock
      continue
    }
    const onGpu = costs.dense[il] + (expertsOnHost ? 0 : costs.experts[il])
    weightsOnGpu += onGpu
    items.push(onGpu + kvPerBlock)
    if (expertsOnHost) hostBytes += costs.experts[il]
  }

  const compute = computeBufferBytes(arch, contextLength, batchSize, flashAttention)
  const logits = logitsBufferBytes(arch, batchSize)

  if (gpuLayers <= 0 || budgets.length === 0) {
    return {
      fits: true,
      perGpu: budgets.map(() => 0),
      layerSplit: budgets.map(() => 0),
      hostBytes: hostBytes + costs.output + compute + logits,
      kvBytes: kvTotal,
      weightsOnGpu: 0,
      overhead: compute
    }
  }

  items.push(costs.output)
  weightsOnGpu += costs.output

  const { counts, perGpu } = placeItems(items, budgets, compute, logits)
  return {
    fits: perGpu.every((need, i) => need <= budgets[i]),
    perGpu,
    layerSplit: counts,
    hostBytes,
    kvBytes: kvTotal,
    weightsOnGpu,
    overhead: compute
  }
}

/*
 * Relative decode speed, from how many bytes each token reads and from where.
 *
 * Generation is bound by memory bandwidth, and system RAM feeds a CPU roughly a tenth as fast as
 * VRAM feeds a GPU. Counting only the bytes a token actually touches is what lets an MoE plan score
 * sensibly: with ten of five hundred experts consulted per token, a block's experts in system RAM
 * cost about a fiftieth of what their size would suggest.
 */
const HOST_SLOWDOWN = 10

function speedScore(arch: ModelArchInfo, gpuLayers: number, cpuMoeLayers: number): number {
  const costs = layerCosts(arch)
  const totalLayers = arch.blockCount
  const used = arch.expertUsedCount ?? 0
  const share = arch.expertCount > 0 && used > 0 ? used / arch.expertCount : 1
  const firstOnGpu = Math.max(0, totalLayers - gpuLayers)
  let gpu = 0
  let host = 0
  for (let il = 0; il < totalLayers; il++) {
    const experts = costs.experts[il] * share
    if (il < firstOnGpu) {
      host += costs.dense[il] + experts
      continue
    }
    gpu += costs.dense[il]
    if (il < cpuMoeLayers) host += experts
    else gpu += experts
  }
  if (gpuLayers > 0) gpu += costs.output
  else host += costs.output
  const total = gpu + host
  if (total <= 0) return 100
  return Math.max(1, Math.round((total / (gpu + host * HOST_SLOWDOWN)) * 100))
}

/**
 * llama.cpp's `-ngl` for a plan.
 *
 * llama.cpp counts its output layer as one more "layer" and offloads from the end, so asking for
 * exactly the block count — which this app did — put the output head on the GPU and left block 0
 * on the CPU: every token took a round trip through system RAM for its first layer. Full offload
 * asks for more layers than any model has, which also covers anything llama.cpp counts beyond
 * block_count (multi-token-prediction layers); a partial offload asks for one more than the blocks
 * it means, for the output layer that goes with them.
 */
export function nglFor(plan: Pick<FitPlan, 'gpuLayers' | 'totalLayers'>): number {
  if (plan.gpuLayers <= 0) return 0
  if (plan.gpuLayers >= plan.totalLayers) return 999
  return plan.gpuLayers + 1
}

/** Largest context that fits with the given layer placement, or 0 if even the minimum fails. */
function maxContextFor(
  arch: ModelArchInfo,
  hw: HardwareSnapshot,
  budgets: number[],
  kvType: KvChoice,
  gpuLayers: number,
  batchSize: number,
  flashAttention: boolean,
  ceiling: number,
  cpuMoeLayers = 0
): number {
  const MIN_CTX = 512
  let lo = 0
  // A non-finite bound would never converge; a caller passing one means "as much as possible".
  let hi = Number.isFinite(ceiling) ? ceiling : 1_048_576

  if (!attempt(arch, hw, budgets, MIN_CTX, kvType, gpuLayers, batchSize, flashAttention, cpuMoeLayers).fits) {
    return 0
  }

  // Binary search on context in 512-token steps.
  lo = MIN_CTX
  while (lo < hi) {
    const mid = Math.min(hi, Math.ceil((lo + hi + 1) / 2 / 512) * 512)
    if (mid === lo) break
    if (attempt(arch, hw, budgets, mid, kvType, gpuLayers, batchSize, flashAttention, cpuMoeLayers).fits) lo = mid
    else hi = mid - 512
  }
  return Math.max(0, Math.floor(lo / 512) * 512)
}

function buildPlan(
  label: string,
  arch: ModelArchInfo,
  hw: HardwareSnapshot,
  budgets: number[],
  contextLength: number,
  kvType: KvChoice,
  gpuLayers: number,
  batchSize: number,
  flashAttention: boolean,
  rationale: string[],
  cpuMoeLayers = 0,
  overrideTensors?: string
): FitPlan {
  const a = attempt(arch, hw, budgets, contextLength, kvType, gpuLayers, batchSize, flashAttention, cpuMoeLayers)
  const totalLayers = arch.blockCount
  const assigned = a.layerSplit.reduce((sum, c) => sum + c, 0)

  const lines = [...rationale]
  /*
   * System RAM is a budget too, once anything lives there.
   *
   * Weights llama.cpp keeps on the host are memory-mapped, so a shortfall does not fail the load:
   * whatever does not fit is read from disk each time it is needed, which on a model spilling tens
   * of gigabytes turns seconds per answer into minutes. Better said before loading than noticed after.
   */
  const freeRam = hw.freeRam ?? 0
  if (freeRam > 0 && a.hostBytes > freeRam * 0.9) {
    lines.push(
      `About ${fmtBytes(a.hostBytes)} has to sit in system RAM and ${fmtBytes(freeRam)} is free. ` +
        'What does not fit will be read from disk as it is needed, which is far slower — close other ' +
        'programs or pick a smaller quantisation.'
    )
  }
  if (overrideTensors) {
    lines.push(
      `Custom tensor overrides are passed to llama.cpp as given (${overrideTensors}); the memory ` +
        'prediction above does not account for them.'
    )
  }

  return {
    label,
    contextLength,
    kvType: kvType.k,
    kvTypeV: kvType.v,
    gpuLayers,
    totalLayers,
    tensorSplit: assigned > 0 ? a.layerSplit.map((c) => c / assigned) : proportionalSplit(budgets),
    layerSplit: a.layerSplit,
    cpuMoeLayers,
    ...(overrideTensors ? { overrideTensors } : {}),
    batchSize,
    flashAttention,
    predictedVramPerGpu: a.perGpu,
    predictedHostBytes: a.hostBytes,
    kvBytes: a.kvBytes,
    weightsOnGpuBytes: a.weightsOnGpu,
    overheadBytes: a.overhead,
    spillsToHost: gpuLayers < totalLayers || cpuMoeLayers > 0,
    speedScore: speedScore(arch, gpuLayers, cpuMoeLayers),
    rationale: lines
  }
}

/**
 * Produce a fit plan (and alternatives when the target cannot be met).
 *
 * Never silently degrades: if we cannot reach targetContext with an acceptable KV type and
 * everything on GPU, we return needsUserChoice with the real tradeoffs rather than picking
 * one quietly. The exception is a mixture-of-experts model, where keeping some routed experts in
 * system RAM is the ordinary way to run it rather than a degradation, and is chosen automatically
 * with the reason stated.
 */
export function planFit(
  arch: ModelArchInfo,
  hw: HardwareSnapshot,
  constraints: FitConstraints = DEFAULT_CONSTRAINTS
): FitResult {
  const notes: string[] = []
  const o = constraints.overrides

  if (!arch.blockCount || !arch.headDim || !arch.headCountKv) {
    return {
      chosen: null,
      alternatives: [],
      needsUserChoice: false,
      hardware: hw,
      notes: ['Model metadata is incomplete (missing layer/head geometry); cannot compute a fit.']
    }
  }

  // Only devices the selected backend can actually address take part in the plan.
  const hasDiscrete = hw.gpus.some((g) => g.totalVram > 0 && !/\bgraphics\b/i.test(g.name))
  const usableGpus = hw.gpus.filter((g) => usableForBackend(g, hw.backend, hasDiscrete))
  const excluded = hw.gpus.filter((g) => !usableGpus.includes(g) && g.totalVram > 0)

  if (excluded.length) {
    notes.push(
      `Excluded from the split: ${excluded.map((g) => g.name).join(', ')} — ` +
        (hw.backend === 'cuda'
          ? 'the CUDA backend can only address NVIDIA devices.'
          : 'integrated GPUs share system memory and would be slower than the discrete cards.')
    )
  }

  const budgets = usableGpus.map((g) => deviceBudget(g, constraints.headroomBytes, hw.backend))

  // The vision encoder loads onto the primary device and brings its own image-processing
  // buffers, so charge it there with margin rather than splitting it across devices.
  const companion = constraints.companionBytes ?? 0
  if (companion > 0 && budgets.length > 0) {
    const charged = Math.max(companion * MMPROJ_OVERHEAD, companion + MMPROJ_MIN_COMPUTE)
    budgets[0] = Math.max(0, budgets[0] - charged)
    notes.push(
      `Reserving ${fmtBytes(charged)} on ${usableGpus[0].name} for the vision projector and its image buffers.`
    )
  }

  const totalBudget = budgets.reduce((a, b) => a + b, 0)

  const anyUnmeasured = usableGpus.some((g) => !g.freeIsMeasured)
  if (anyUnmeasured) {
    notes.push(
      `Free VRAM could not be measured on ${usableGpus.filter((g) => !g.freeIsMeasured).map((g) => g.name).join(', ')}; ` +
        `assuming ${Math.round(UNMEASURED_FREE_FRACTION * 100)}% of total is available.`
    )
  }
  if (usableGpus.length > 1) {
    const pct = proportionalSplit(budgets).map((s) => `${Math.round(s * 100)}%`)
    notes.push(
      `Split across ${usableGpus.length} GPUs by free capacity: ${pct.join(' / ')} — proportional, not even.`
    )
  }

  // Honour user overrides exactly — they are respected, never silently changed.
  const batchSize = o.batchSize ?? 512
  const flashAttention = o.flashAttention ?? true
  if (!flashAttention) {
    notes.push(
      'Flash attention is off. The attention buffer grows with batch x context x heads, which ' +
        'dominates VRAM at long context — expect a much smaller achievable context.'
    )
  }

  /*
   * The trained length is the ceiling unless someone deliberately lifts it.
   *
   * Past it the model is being handed positions it never saw, which llama.cpp can do with YaRN
   * and which the long-context benchmarks suggest is a poorer deal than it sounds: most models
   * already fall below their advertised effective length well before reaching it. So the default
   * stops here, and `allowRopeScaling` is what says otherwise.
   */
  const trainedCeiling = arch.contextLength > 0 ? arch.contextLength : constraints.idealContext
  const hardCeiling = constraints.allowRopeScaling ? trainedCeiling * MAX_ROPE_SCALE : trainedCeiling
  const ceiling = Math.min(o.contextLength ?? constraints.idealContext, hardCeiling, constraints.idealContext)

  if (arch.contextLength > 0 && ceiling >= arch.contextLength && constraints.idealContext > arch.contextLength) {
    notes.push(
      constraints.allowRopeScaling
        ? `Model was trained for ${arch.contextLength.toLocaleString()} tokens. Rope scaling is enabled, so more than that may be planned — quality past the trained length is not guaranteed.`
        : `Model was trained for ${arch.contextLength.toLocaleString()} tokens; capping there.`
    )
  }

  if (arch.ssmLayers > 0) {
    notes.push(
      `Hybrid architecture: only ${arch.attentionLayers} of ${arch.blockCount} layers hold a KV cache ` +
        `(the other ${arch.ssmLayers} are state-space layers with a fixed ` +
        `${fmtBytes(arch.ssmLayers * arch.ssmStateBytesPerLayer)} state). Long context is far cheaper here than layer count suggests.`
    )
  }

  const totalLayers = arch.blockCount
  const fullGpuLayers = o.gpuLayers ?? totalLayers

  /*
   * Mixture-of-experts: what can move to system RAM before whole layers have to.
   *
   * Most of an MoE model is routed experts, and only a handful are read for any one token — so
   * keeping some of them in system RAM costs a small fraction of the speed that moving the same
   * bytes of attention or shared weights would. llama.cpp's `--n-cpu-moe` does exactly that for
   * the first N blocks. A user's explicit N is honoured as given, everywhere below.
   */
  const costs = layerCosts(arch)
  const experts = expertSpan(costs)
  const cpuMoeFixed =
    o.cpuMoeLayers !== undefined ? Math.max(0, Math.min(Math.round(o.cpuMoeLayers), totalLayers)) : undefined
  const cpuMoeBase = cpuMoeFixed ?? 0
  const overrideTensors = o.overrideTensors?.trim() || undefined

  if ((arch.pleBytes ?? 0) > 0) {
    notes.push(
      `The ${fmtBytes(arch.pleBytes ?? 0)} per-layer embedding table stays in system RAM, where llama.cpp ` +
        'always keeps it: it is only ever looked up a few rows per token, so a GPU would hold a copy and add nothing.'
    )
  }

  if (totalBudget <= 0) {
    const plan = buildPlan('CPU only', arch, hw, budgets, Math.min(8192, ceiling), { k: constraints.preferredKvType, v: constraints.preferredKvType }, 0, batchSize, flashAttention, [
      'No usable GPU memory detected — running entirely on CPU.'
    ], 0, overrideTensors)
    return { chosen: plan, alternatives: [], needsUserChoice: false, hardware: hw, notes }
  }

  /*
   * Candidate KV types, best-quality first, never below the configured floor.
   *
   * The two settings are independent dropdowns, and their lists are ordered opposite ways, so
   * "prefer q4_0, floor q8_0" is an easy thing to end up with — a contradiction, since q4_0 is
   * below the floor it is supposedly preferred over. A plain `slice(preferred, floor + 1)`
   * returns an *empty* list for that, and an empty list means both passes that can return a
   * chosen plan iterate zero times: auto-fit quietly stops auto-fitting, every model in the
   * library reports that it needs a manual decision, and nothing says why.
   *
   * An unrecognised value — a hand-edited settings.json — has the same shape of problem, with
   * `indexOf` returning -1 and `slice(-1, …)` silently yielding the worst type regardless of
   * what was asked for.
   *
   * The floor wins where the two disagree, because a floor is the stronger statement: it says
   * what is unacceptable, where the preference only says what is nicest.
   */
  const floorIndex = clampToOrder(constraints.minKvType, KV_ORDER.length - 1)
  const kvCandidates: KvChoice[] = o.kvType
    ? [{ k: o.kvType, v: o.kvTypeV ?? o.kvType }]
    : kvLadder(constraints.preferredKvType, constraints.minKvType)

  /*
   * With everything on the GPU, an MoE model does not step its keys down to buy context.
   *
   * Four-bit keys are the one cache setting measured to break a model rather than blunt it, and
   * the ladder only reaches them when nothing else fits. An MoE model always has something else:
   * its experts can move to system RAM instead, which costs speed, not answers. So for those the
   * full-GPU passes stop at the preferred key precision and leave the rest to pass 3.
   */
  const fullGpuCandidates =
    experts.blocks > 0 && cpuMoeFixed === undefined && !o.kvType
      ? kvCandidates.filter((c) => c.k === kvCandidates[0]?.k)
      : kvCandidates

  const warnIfKeysPushed = (kv: KvChoice, into: string[]): void => {
    if (kv.k === 'q4_0') {
      into.push(
        'Keys are at q4_0, which is the bottom of the ladder and the one step that can cost real ' +
          'accuracy rather than a rounding error. Reduce the context or raise the KV floor in ' +
          'Settings if answers look wrong.'
      )
    }
  }

  if (KV_ORDER.indexOf(constraints.preferredKvType) > floorIndex) {
    notes.push(
      `Preferred KV (${constraints.preferredKvType}) is below the floor (${constraints.minKvType}), ` +
        `so the floor was used. Lower the floor in Settings if you want ${constraints.preferredKvType}.`
    )
  }

  /*
   * Pass 1: the best cache quality that clears the target, and as much context as that allows.
   *
   * The order of these two questions is the whole behaviour. Asking "which candidate reaches the
   * largest context" first means the answer is always the coarsest cache, because that is what
   * being coarse buys — on a hybrid 27B it planned the full 262,144 tokens with four-bit keys,
   * which is the configuration measured to break a model. Asking "which candidate is good
   * enough" first, and then spending whatever it leaves on context, gives up some length and
   * keeps the answers.
   *
   * Quality is only traded away when the target cannot be met at all, which is pass 2.
   */
  for (const kvType of fullGpuCandidates) {
    const maxCtx = maxContextFor(arch, hw, budgets, kvType, fullGpuLayers, batchSize, flashAttention, ceiling, cpuMoeBase)
    /*
     * Clamped to the ceiling, which the target is not bound by.
     *
     * `targetContext` is what the user asked for and the ceiling is what the model or an explicit
     * override permits; taking the larger of target and ideal without clamping let a 64K target
     * override a 32K request.
     */
    const enough = Math.min(ceiling, Math.max(constraints.targetContext, Math.min(constraints.idealContext, ceiling)))
    /*
     * The target, but never more than the ceiling allows.
     *
     * A model trained to 8K, or an explicit 8K override, cannot clear a 64K target however much
     * VRAM is free — and treating that as a failure sent a perfectly good full-GPU plan down the
     * "present the user with tradeoffs" path for no reason. Reaching the ceiling is success.
     */
    if (maxCtx >= Math.min(constraints.targetContext, ceiling)) {
      const ctx = Math.min(enough, maxCtx)
      const reachedIdeal = ctx >= Math.min(constraints.idealContext, ceiling)
      const rationale = [
        cpuMoeBase > 0
          ? `All ${totalLayers} layers on GPU, with the routed experts of the first ${cpuMoeBase} kept in system RAM as set.`
          : `All ${totalLayers} layers on GPU.`,
        reachedIdeal
          ? `KV cache at ${kvLabel(kvType)} reaches ${ctx.toLocaleString()} tokens — the ideal target.`
          : `KV cache at ${kvLabel(kvType)} reaches ${ctx.toLocaleString()} tokens, the most this quality affords.`,
        'Full context KV is reserved up front, so the load cannot OOM as the chat grows.'
      ]
      if (!reachedIdeal) {
        rationale.push(
          'A coarser cache would fit more context; it is not used because the extra length is ' +
            'worth less than the accuracy it would cost.'
        )
      }
      warnIfKeysPushed(kvType, rationale)

      /*
       * Offer the longer context, without taking it.
       *
       * The values tolerate a coarser cache in a way the keys do not — llama.cpp's own
       * comparisons put q4_0 values at about one changed answer in five hundred, against a
       * collapse for q4_0 keys. So there is usually a rung below the chosen one that buys real
       * length for very little, and the right thing is to put it in front of the user rather
       * than to decide for them: this is exactly the trade someone feeding it long documents or
       * video wants to make, and someone doing careful work does not.
       *
       * Only ever with the keys left where they are. A rung that touches them is a different
       * kind of offer and does not belong beside this one.
       */
      const longer: FitPlan[] = []
      for (const other of kvCandidates.slice(kvCandidates.indexOf(kvType) + 1)) {
        if (other.k !== kvType.k) continue
        const otherMax = maxContextFor(arch, hw, budgets, other, fullGpuLayers, batchSize, flashAttention, ceiling, cpuMoeBase)
        // Worth showing only if it is a step, not a rounding difference.
        if (otherMax < ctx * 1.1) continue
        longer.push(
          buildPlan('More context', arch, hw, budgets, Math.min(otherMax, ceiling), other, fullGpuLayers, batchSize, flashAttention, [
            `All ${totalLayers} layers on GPU.`,
            `Values at ${other.v} instead of ${kvType.v} reach ${Math.min(otherMax, ceiling).toLocaleString()} tokens — ` +
              `${Math.round((Math.min(otherMax, ceiling) / ctx - 1) * 100)}% more than the default plan.`,
            'Keys are unchanged, which is the half that carries the accuracy; coarser values ' +
              'measure at roughly one changed answer in five hundred.'
          ], cpuMoeBase, overrideTensors)
        )
        break
      }

      return {
        chosen: buildPlan(
          reachedIdeal ? 'Ideal' : 'Best quality',
          arch, hw, budgets, ctx, kvType, fullGpuLayers, batchSize, flashAttention, rationale, cpuMoeBase, overrideTensors
        ),
        alternatives: longer,
        needsUserChoice: false,
        hardware: hw,
        notes
      }
    }
  }

  // Pass 2: everything on GPU, accept the best context we can reach at or above target.
  for (const kvType of fullGpuCandidates) {
    const maxCtx = maxContextFor(arch, hw, budgets, kvType, fullGpuLayers, batchSize, flashAttention, ceiling, cpuMoeBase)
    if (maxCtx >= Math.min(constraints.targetContext, ceiling)) {
      const rationale = [
        `All ${totalLayers} layers on GPU.`,
        `KV at ${kvLabel(kvType)} reaches ${maxCtx.toLocaleString()} tokens, above the ${constraints.targetContext.toLocaleString()} target.`,
        kvType.k !== constraints.preferredKvType || kvType.v !== constraints.preferredKvType
          ? `Stepped down from ${constraints.preferredKvType} to ${kvLabel(kvType)} to buy context, keeping the keys at or above the ${constraints.minKvType} floor.`
          : 'KV kept at the preferred quality.'
      ]
      return {
        chosen: buildPlan('Balanced', arch, hw, budgets, maxCtx, kvType, fullGpuLayers, batchSize, flashAttention, rationale, cpuMoeBase, overrideTensors),
        alternatives: [],
        needsUserChoice: false,
        hardware: hw,
        notes
      }
    }
  }

  /*
   * Pass 3: an MoE model that fits once some of its experts are in system RAM.
   *
   * Every block stays on the GPU — attention, shared expert, norms — and routed experts move to
   * the host a block at a time from the front, only as many as it takes to reach the target
   * context. That is llama.cpp's `--n-cpu-moe`, with N found for this machine rather than guessed.
   * The cache ladder is walked best-first, as in pass 1: the first rung that reaches the target
   * with some N wins, and whatever VRAM that leaves goes to context.
   */
  if (experts.blocks > 0 && cpuMoeFixed === undefined && o.gpuLayers === undefined) {
    const targetCtx = Math.min(constraints.targetContext, ceiling)
    for (const kvType of kvCandidates) {
      if (!attempt(arch, hw, budgets, targetCtx, kvType, totalLayers, batchSize, flashAttention, experts.end).fits) continue

      // The fewest blocks whose experts have to leave: each one more costs speed, so find the edge.
      let lo = 1
      let hi = experts.end
      while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2)
        if (attempt(arch, hw, budgets, targetCtx, kvType, totalLayers, batchSize, flashAttention, mid).fits) hi = mid
        else lo = mid + 1
      }
      const cpuMoe = lo

      const enough = Math.min(ceiling, Math.max(constraints.targetContext, Math.min(constraints.idealContext, ceiling)))
      const ctx = Math.min(
        enough,
        maxContextFor(arch, hw, budgets, kvType, totalLayers, batchSize, flashAttention, ceiling, cpuMoe)
      )
      const moved = costs.experts.slice(0, cpuMoe)
      const movedBlocks = moved.filter((b) => b > 0).length
      const movedBytes = moved.reduce((a, b) => a + b, 0)
      const expertTotal = costs.experts.reduce((a, b) => a + b, 0)
      const rationale = [
        `All ${totalLayers} layers on GPU, with the routed experts of ${movedBlocks} of ${experts.blocks} expert ` +
          `layers kept in system RAM (${fmtBytes(movedBytes)} of ${fmtBytes(expertTotal)}).`,
        (arch.expertUsedCount ?? 0) > 0 && arch.expertCount > 0
          ? `Attention, the shared expert and the output head stay on the GPU. Only ${arch.expertUsedCount} of ` +
            `${arch.expertCount} experts are read for each token, so experts in system RAM cost far less speed than their size suggests.`
          : 'Attention, the shared expert and the output head stay on the GPU, which is where every token needs them.',
        `KV cache at ${kvLabel(kvType)} reaches ${ctx.toLocaleString()} tokens.`,
        'Full context KV is reserved up front, so the load cannot OOM as the chat grows.'
      ]
      warnIfKeysPushed(kvType, rationale)

      /*
       * Offer the rest of the context, at the price of more experts in system RAM.
       *
       * With every block's experts on the host, what is left of VRAM goes to the cache — slower
       * generation for a longer window, which is the trade someone feeding it long documents wants
       * to be offered rather than to have made for them.
       */
      const alternatives: FitPlan[] = []
      const allOut = maxContextFor(arch, hw, budgets, kvType, totalLayers, batchSize, flashAttention, ceiling, experts.end)
      if (experts.end > cpuMoe && allOut >= ctx * 1.1) {
        const longer = Math.min(allOut, ceiling)
        alternatives.push(
          buildPlan('More context', arch, hw, budgets, longer, kvType, totalLayers, batchSize, flashAttention, [
            `All ${totalLayers} layers on GPU, with every routed expert in system RAM (${fmtBytes(expertTotal)}).`,
            `KV cache at ${kvLabel(kvType)} reaches ${longer.toLocaleString()} tokens — ` +
              `${Math.round((longer / ctx - 1) * 100)}% more than the default plan, for slower generation.`
          ], experts.end, overrideTensors)
        )
      }

      return {
        chosen: buildPlan(
          'Experts in RAM', arch, hw, budgets, ctx, kvType, totalLayers, batchSize, flashAttention, rationale, cpuMoe, overrideTensors
        ),
        alternatives,
        needsUserChoice: false,
        hardware: hw,
        notes
      }
    }
  }

  // Target unreachable with everything on GPU. Present real tradeoffs instead of choosing.
  const alternatives: FitPlan[] = []
  /*
   * The bottom of the ladder: keys at the floor, values as far down as they go.
   *
   * This is the last configuration tried before layers start moving to the host, so it should be
   * the smallest one that is still safe rather than the smallest one that exists.
   */
  const floorKv: KvChoice = kvLadder(constraints.preferredKvType, constraints.minKvType).at(-1) ?? {
    k: constraints.minKvType,
    v: constraints.minKvType
  }
  // An MoE model's experts all leave the GPU before any whole block does.
  const cpuMoeAlt = cpuMoeFixed ?? experts.end
  const expertsNote = cpuMoeAlt > 0 ? ', with every routed expert in system RAM' : ''

  // (a) Keep the target context, offload layers to host.
  let layersForTarget = 0
  for (let l = totalLayers; l >= 0; l--) {
    if (attempt(arch, hw, budgets, constraints.targetContext, floorKv, l, batchSize, flashAttention, cpuMoeAlt).fits) {
      layersForTarget = l
      break
    }
  }
  if (layersForTarget > 0 || totalLayers === 0) {
    alternatives.push(
      buildPlan(
        `Keep ${constraints.targetContext.toLocaleString()} context`,
        arch, hw, budgets, constraints.targetContext, floorKv, layersForTarget, batchSize, flashAttention,
        [
          `${layersForTarget} of ${totalLayers} layers on GPU${expertsNote}, the rest on CPU.`,
          `Preserves the ${constraints.targetContext.toLocaleString()}-token context you asked for.`,
          'Generation will be slower in proportion to the layers running on CPU.'
        ],
        cpuMoeAlt,
        overrideTensors
      )
    )
  }

  // (b) Keep all layers on GPU, shrink context.
  const maxCtxFloor = maxContextFor(arch, hw, budgets, floorKv, fullGpuLayers, batchSize, flashAttention, ceiling, cpuMoeAlt)
  if (maxCtxFloor > 0) {
    alternatives.push(
      buildPlan(
        cpuMoeAlt > 0 ? 'Shorter context' : 'Max speed',
        arch, hw, budgets, maxCtxFloor, floorKv, fullGpuLayers, batchSize, flashAttention,
        [
          cpuMoeAlt > 0
            ? `All ${totalLayers} layers on GPU${expertsNote}.`
            : `All ${totalLayers} layers on GPU — fastest generation.`,
          `Context limited to ${maxCtxFloor.toLocaleString()} tokens, below your ${constraints.targetContext.toLocaleString()} target.`,
          `KV at ${kvLabel(floorKv)} (the floor) to stretch context as far as possible.`
        ],
        cpuMoeAlt,
        overrideTensors
      )
    )
  }

  // (c) A smaller quant would fix the root cause rather than degrading the load.
  const gpuNeeded =
    experts.blocks > 0
      ? costs.dense.reduce((a, b) => a + b, 0) + costs.output
      : Math.max(0, arch.weightBytes - costs.input)
  const deficit = gpuNeeded + kvCacheBytes(arch, constraints.targetContext, floorKv) - totalBudget
  if (deficit > 0) {
    notes.push(
      `This model needs roughly ${fmtBytes(deficit)} more VRAM to hit ${constraints.targetContext.toLocaleString()} tokens ` +
        (experts.blocks > 0
          ? 'even with every routed expert in system RAM. A smaller quantisation of the same model would fit properly.'
          : 'with everything on GPU. A smaller quantisation of the same model would fit properly.')
    )
  }

  alternatives.sort((a, b) => b.speedScore - a.speedScore)

  return {
    chosen: null,
    alternatives,
    needsUserChoice: true,
    hardware: hw,
    notes
  }
}

export function fmtBytes(n: number): string {
  if (!Number.isFinite(n)) return '?'
  // Mirrors the renderer's formatter, including the kilobyte step: without it anything under a
  // megabyte printed as a raw byte count.
  const units: [number, string, number][] = [
    [1024 ** 4, 'TB', 2],
    [GB, 'GB', 2],
    [MB, 'MB', 0],
    [1024, 'KB', 0]
  ]
  for (const [size, label, digits] of units) {
    if (n >= size) {
      const value = n / size
      return `${value.toFixed(value >= 10 ? Math.max(0, digits - 1) : digits)} ${label}`
    }
  }
  return `${Math.round(n)} B`
}

/**
 * Compare prediction against reality after a load and report the delta.
 * Persistent error feeds back into the headroom margin so the engine gets more accurate
 * on each specific machine over time.
 */
export function verifyPrediction(plan: FitPlan, actualPerGpu: number[]): {
  deltas: number[]
  worstRatio: number
  suggestion: string | null
} {
  const deltas = plan.predictedVramPerGpu.map((p, i) => (actualPerGpu[i] ?? 0) - p)
  const ratios = plan.predictedVramPerGpu.map((p, i) => (p > 0 ? (actualPerGpu[i] ?? 0) / p : 1))
  const worstRatio = ratios.length ? Math.max(...ratios) : 1

  // A reading of zero means the measurement failed, not that the model used no memory —
  // driver hiccup, a device we cannot poll, or a stand-in server during testing. Reporting
  // "100% below prediction" from that is worse than saying nothing.
  const measured = actualPerGpu.filter((v) => v > 0)
  if (measured.length === 0) {
    return { deltas, worstRatio, suggestion: null }
  }

  let suggestion: string | null = null
  if (worstRatio > 1.1) {
    const over = fmtBytes(Math.max(...deltas))
    suggestion = `Used ${over} more VRAM than predicted (${Math.round((worstRatio - 1) * 100)}% over). Widening the safety margin for next time.`
  } else if (worstRatio < 0.85) {
    const spare = fmtBytes(Math.abs(Math.min(...deltas)))
    suggestion = `Used ${spare} less VRAM than predicted. There is room to raise the context on this model.`
  }
  return { deltas, worstRatio, suggestion }
}
