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
  ModelArchInfo,
  PlanCorrection,
  PlanDevice,
  VramSegments
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
 *
 * Pairings the backend cannot attend over directly are left out when it names the ones it can. A
 * CUDA build converts the whole cache to f16 on every generated token for a pairing it did not
 * compile, which spends more speed and VRAM than the smaller cache saves. A list naming none of
 * the rungs leaves the ladder as it was: a slow plan beats no plan.
 */
function kvLadder(preferred: KvType, floor: KvType, fastPairs?: readonly string[]): KvChoice[] {
  const floorIndex = clampToOrder(floor, KV_ORDER.length - 1)
  const preferredIndex = Math.min(clampToOrder(preferred, 0), floorIndex)
  const pairs: KvChoice[] = []
  for (const k of KV_ORDER.slice(preferredIndex, floorIndex + 1)) {
    // Values never finer than the keys: spending more on the tolerant half would be backwards.
    for (const v of KV_ORDER.slice(KV_ORDER.indexOf(k))) pairs.push({ k, v })
  }
  const size = (c: KvChoice): number => KV_ELEMENT_BYTES[c.k] + KV_ELEMENT_BYTES[c.v]
  pairs.sort((a, b) => size(b) - size(a))
  if (!fastPairs) return pairs
  const fast = pairs.filter((c) => fastPairs.includes(kvPairName(c)))
  return fast.length ? fast : pairs
}

/** A pairing as llama.cpp's build features name it: keys first, as in `q8_0-q4_0`. */
function kvPairName(kv: KvChoice): string {
  return `${kv.k}-${kv.v}`
}

/** CUDA runtime + cuBLAS workspace claimed per device before any weights load (measured 0.39 GB). */
const CUDA_CONTEXT_OVERHEAD = 400 * MB
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

/**
 * Kept free beyond the user's reserve when nothing about the plan has been measured.
 *
 * Small, because what it guards against is allocator slack and a few hundred megabytes of driver
 * variance, not errors in the estimate: those are measured away. Plans llama.cpp has confirmed
 * use a smaller figure; see the planner.
 */
export const DEFAULT_SAFETY = 256 * MB

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
  /*
   * A sparse-attention indexer caches on the same layers: one key per token at the key precision,
   * and one pooled f32 key per block, which llama.cpp keeps so the indexer does not re-pool the
   * whole context on every token. Qwen3.8-Flash-Next at 199,680 tokens: about 630 MB together.
   */
  const indexerKey = arch.indexerKeyLength ?? 0
  const indexerPerToken =
    indexerKey > 0
      ? indexerKey * KV_ELEMENT_BYTES[choice.k] + (arch.indexerBlock ? (indexerKey * 4) / arch.indexerBlock : 0)
      : 0
  const indexerCache = indexerPerToken * attentionLayers * contextLength
  return attentionCache + recurrentState + indexerCache
}

/**
 * Each block's share of the cache, in block order.
 *
 * Only attention blocks cache per token; a hybrid model's state-space blocks hold a fixed state.
 * Spreading the whole cache evenly over every block put a twelfth of an attention layer on each
 * of them, so which card got which blocks moved half a gigabyte between cards on a 500K context
 * that the estimate could not see. llama.cpp's hybrid layout puts attention on every n-th block,
 * the last of each group (block i when (i + 1) % n == 0), with n the ratio of blocks to attention
 * blocks, which is how this reads it without the interval being stored.
 */
export function kvPerLayer(arch: ModelArchInfo, contextLength: number, kv: KvType | KvChoice): number[] {
  const n = arch.blockCount
  const total = kvCacheBytes(arch, contextLength, kv)
  const ssm = arch.ssmLayers ?? 0
  const attention = arch.attentionLayers || n
  if (!(ssm > 0 && attention > 0 && attention < n)) return new Array<number>(n).fill(n > 0 ? total / n : 0)
  const interval = Math.max(2, Math.round(n / attention))
  const state = (arch.ssmStateBytesPerLayer ?? 0) * SSM_STATE_SAFETY
  const isAttention = (il: number): boolean => (il + 1) % interval === 0
  const attentionCount = Array.from({ length: n }, (_, il) => il).filter(isAttention).length || 1
  const perAttention = Math.max(0, total - ssm * state) / attentionCount
  return Array.from({ length: n }, (_, il) => (isAttention(il) ? perAttention : state))
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
  /*
   * A sparse-attention indexer scores every cached token for every token of the batch, whatever
   * flash attention does for the main heads. On Qwen3.8-Flash-Next at ubatch 2048 the compute
   * buffer on one card grew from 1.5 GB at 64K to 3.7 GB at 500K: about 2.7 bytes per
   * (batch token x context token). Measured plans replace this with llama.cpp's own figure.
   */
  const indexer = (arch.indexerKeyLength ?? 0) > 0 ? batchSize * contextLength * 2.7 : 0
  return activations + attention + indexer + BASE_GRAPH_OVERHEAD
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
 * Compute-buffer cost of one block whose routed experts run on the CPU.
 *
 * A block placed on a GPU with its experts in system RAM has to hand each token's hidden state
 * across to the host and take the result back, and llama.cpp reserves room for that in the
 * compute buffer of the device holding the block — so the cost scales with how many such blocks
 * a card ends up with, which is why it is charged per block rather than per device.
 *
 * Measured on Qwen3.8-Flash-Next at batch 512: 1,913 MiB of compute buffer on a card holding 43
 * such blocks against 527 MiB on one holding 5, which is 36.5 MiB per block over a fixed base.
 * It does not grow in step with the batch: with all 48 blocks' experts in RAM the two cards'
 * compute buffers measured 1.9 GiB at batch 512 and 3.8 GiB at 2048, where seven copies per token
 * predicted 7 GB of staging alone. Fitted to both, rounded up: about 36.6 MiB per block at 512 and
 * 64 MiB at 2048, in hidden-state copies so it follows the embedding width to other models.
 */
function hostExpertStaging(arch: ModelArchInfo, batchSize: number): number {
  if (arch.expertCount <= 0) return 0
  return (2800 + 1.85 * batchSize) * arch.embeddingLength * 4
}

/**
 * What each participating device needs beyond the graph estimate above.
 *
 * Measured, not derived: a Qwen3.8-Flash-Next load that the planner sized at 12.3 GB against
 * 13.4 GB of free VRAM died twice — once on the KV cache, once on the recurrent-state cache with
 * 103 MiB left to find. Recurrent models allocate that state cache per device on top of the
 * graph, and a routed model stages expert rows through buffers this estimate does not model at
 * all. Both are charged here so a plan leaves room for them instead of discovering them at load.
 *
 * Erring high costs a little context or one expert layer. Erring low costs the whole load, which
 * is why these round up rather than down.
 */
function deviceReserve(arch: ModelArchInfo): number {
  const recurrent = (arch.ssmLayers ?? 0) > 0 ? 256 * MB : 0
  const routed = arch.expertCount > 0 ? 384 * MB : 0
  return recurrent + routed
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

/**
 * The devices a plan covers, in the order of its per-card figures (`predictedVramPerGpu`).
 *
 * Exported so whoever measures a load reads the same cards in the same order: the hardware list
 * also holds devices the backend cannot use, such as an integrated GPU under CUDA.
 */
export function usableDevices(hw: HardwareSnapshot): GpuDevice[] {
  const hasDiscrete = hw.gpus.some((g) => g.totalVram > 0 && !/\bgraphics\b/i.test(g.name))
  return hw.gpus.filter((g) => usableForBackend(g, hw.backend, hasDiscrete))
}

/** A card's name that survives a restart: its position and model, so two identical cards stay apart. */
export function gpuKey(gpu: GpuDevice): string {
  return `${gpu.index}:${gpu.name}`
}

/**
 * Usable bytes on a device: what is free, less what is kept free on purpose, less what goes to
 * things outside llama.cpp's own buffers.
 *
 * There is no hidden margin beyond these. Each used to be a separate guess (a fixed runtime
 * overhead, a fraction of the card for routed models, a learned blob of "overshoot") and together
 * they left gigabytes unused on cards where the user had asked for 512 MB to be kept free.
 */
function deviceBudget(gpu: GpuDevice, keepFree: number, extras: number): number {
  // P1: prefer the measured free figure; fall back to a discounted total only when we
  // genuinely have no way to measure, and never to the raw total.
  const free = gpu.freeIsMeasured && gpu.freeVram >= 0 ? gpu.freeVram : gpu.totalVram * UNMEASURED_FREE_FRACTION
  return Math.max(0, free - keepFree - extras)
}

/**
 * The extras each card is charged before anything was measured on this machine.
 *
 * The CUDA runtime on every card; the vision projector and a separate MTP head on the first, which
 * is where llama-server puts them. Fitted to Qwen3.8-Flash-Next's 500K load on an RTX 5080 and
 * 4070 Ti: 0.39 GB of runtime per card, a 0.9 GB projector taking 1.7 GB with its encoder buffers.
 * Replaced by the measured figure after the model's first load.
 */
export function priorExtras(cards: number, backend: string, companionBytes: number, draftBytes: number): number[] {
  const runtime = backend === 'cuda' ? CUDA_CONTEXT_OVERHEAD : 0
  const projector = companionBytes > 0 ? companionBytes + Math.max(MMPROJ_MIN_COMPUTE, companionBytes * 0.9) : 0
  return Array.from({ length: cards }, (_, d) => runtime + (d === 0 ? projector + draftBytes : 0))
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
  /** The expert cache's share of `perGpu`. */
  cachePerGpu: number[]
  /** `perGpu` split by kind; the parts of each card add up to its figure there. */
  segments: VramSegments[]
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

  /*
   * Two cards: try every cut and keep the one that leaves the most room on the tighter card.
   *
   * A proportional cut is a fair first guess and nothing more: it left 0.85 GB spare on an RTX 5080
   * beside 0.37 GB on a 4070 Ti, room the expert cache could have used. Forty-odd cuts cost nothing.
   */
  if (devices === 2 && items.length > 1) {
    let best: { counts: number[]; perGpu: number[]; slack: number } | null = null
    for (let k = 0; k <= items.length; k++) {
      const counts = [k, items.length - k]
      const need = measure(counts)
      const slack = Math.min(budgets[0] - need[0], budgets[1] - need[1])
      if (!best || slack > best.slack) best = { counts, perGpu: need, slack }
    }
    if (best) return { counts: best.counts, perGpu: best.perGpu }
  }

  const even = cutProportional(items, caps)
  const evenNeed = measure(even)
  if (devices === 1 || fitsAll(evenNeed)) return { counts: even, perGpu: evenNeed }

  const filled = cutGreedy(items, caps)
  const filledNeed = measure(filled)
  return fitsAll(filledNeed) ? { counts: filled, perGpu: filledNeed } : { counts: even, perGpu: evenNeed }
}

/**
 * What each card holds, by kind: weights, cache, and working memory.
 *
 * Follows `placeItems`'s accounting exactly — the same cut, a compute buffer on every card holding
 * anything, the logits on the card with the output layer — so the parts always add up to the card's
 * prediction and a bar drawn from them cannot disagree with the number beside it.
 */
function deviceSegments(
  counts: number[],
  weights: number[],
  kv: number[],
  staging: number[],
  compute: number,
  logits: number,
  cache: number[]
): VramSegments[] {
  const out = counts.map(() => ({ weights: 0, kv: 0, compute: 0, cache: 0, extras: 0 }))
  let k = 0
  let outputDevice = 0
  counts.forEach((n, d) => {
    for (let j = 0; j < n; j++, k++) {
      out[d].weights += weights[k] - cache[k]
      out[d].cache += cache[k]
      out[d].kv += kv[k]
      out[d].compute += staging[k]
    }
    if (n > 0) {
      out[d].compute += compute
      outputDevice = d
    }
  })
  if (out.length) out[outputDevice].compute += logits
  return out
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
/*
 * GPU expert-cache slots per host-expert layer for the plan being computed.
 *
 * Set by planFit for the length of one synchronous planning call rather than threaded through every
 * placement helper. llama.cpp puts a layer's cache beside its router, so it is charged to the card
 * that holds the layer: (slots + 1) of the layer's experts, the extra one being the zero slot.
 */
let planExpertCacheSlots = 0

/*
 * The rest of one planning call's context, set by planFit for its duration like the slots above:
 * measured corrections to apply, the extras each card carries, and what each card keeps free.
 */
let planCorrections: PlanCorrection[] = []
let planExtras: number[] = []
let planKeepFree: number[] = []

/**
 * How far llama.cpp's own figure differs from the estimate, per card, for a configuration.
 *
 * Taken from measurements of the same cache types, batch and placement, interpolated across
 * context: the difference is nearly all compute buffer, which tracks context and not much else.
 * When nothing matching has been measured, the nearest measurement with the same batch is used,
 * and when there is none, nothing is corrected.
 */
export function correctionFor(
  corrections: readonly PlanCorrection[],
  q: { contextLength: number; kvType: KvChoice; ubatch: number; flashAttention: boolean; gpuLayers: number; cpuMoeLayers: number },
  cards: number
): number[] {
  const zero = new Array<number>(cards).fill(0)
  const sameBatch = corrections.filter(
    (c) => c.ubatch === q.ubatch && c.flashAttention === q.flashAttention && c.offset.length === cards
  )
  if (!sameBatch.length) return zero
  const group = sameBatch.filter(
    (c) =>
      c.kvType === q.kvType.k &&
      c.kvTypeV === q.kvType.v &&
      c.gpuLayers === q.gpuLayers &&
      c.cpuMoeLayers === q.cpuMoeLayers
  )
  const pool = group.length ? group : sameBatch
  const near = [...pool].sort(
    (a, b) => Math.abs(a.contextLength - q.contextLength) - Math.abs(b.contextLength - q.contextLength)
  )[0]
  if (!group.length) return near.offset
  const below = group.filter((c) => c.contextLength <= q.contextLength).sort((a, b) => b.contextLength - a.contextLength)[0]
  const above = group.filter((c) => c.contextLength >= q.contextLength).sort((a, b) => a.contextLength - b.contextLength)[0]
  if (below && above && above.contextLength !== below.contextLength) {
    const t = (q.contextLength - below.contextLength) / (above.contextLength - below.contextLength)
    return below.offset.map((o, d) => o + (above.offset[d] - o) * t)
  }
  // One side only: carry the nearest offset across, which the next measurement then corrects.
  return near.offset
}

function expertCacheBytes(arch: ModelArchInfo, expertsBytes: number): number {
  if (planExpertCacheSlots <= 0 || arch.expertCount <= 0) return 0
  return ((planExpertCacheSlots + 1) * expertsBytes) / arch.expertCount
}

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
  const kvByBlock = kvPerLayer(arch, contextLength, kvType)
  const firstOnGpu = Math.max(0, totalLayers - gpuLayers)
  const cpuMoe = Math.max(0, Math.min(cpuMoeLayers, totalLayers))

  let hostBytes = costs.input
  let weightsOnGpu = 0
  const items: number[] = []
  // The same items split by kind, so each card's share can be shown as weights, cache and working memory.
  const itemWeights: number[] = []
  const itemKv: number[] = []
  const itemStaging: number[] = []
  const itemCache: number[] = []
  for (let il = 0; il < totalLayers; il++) {
    const expertsOnHost = il < cpuMoe
    if (il < firstOnGpu) {
      hostBytes += costs.dense[il] + costs.experts[il] + kvByBlock[il]
      continue
    }
    const cacheBytes = expertsOnHost ? expertCacheBytes(arch, costs.experts[il]) : 0
    const onGpu = costs.dense[il] + (expertsOnHost ? cacheBytes : costs.experts[il])
    // Staging travels with the block, because that is where llama.cpp allocates it.
    const staging = expertsOnHost ? hostExpertStaging(arch, batchSize) : 0
    weightsOnGpu += onGpu
    items.push(onGpu + kvByBlock[il] + staging)
    itemWeights.push(onGpu)
    itemKv.push(kvByBlock[il])
    itemStaging.push(staging)
    itemCache.push(cacheBytes)
    if (expertsOnHost) hostBytes += costs.experts[il]
  }

  const compute = computeBufferBytes(arch, contextLength, batchSize, flashAttention) + deviceReserve(arch)
  const logits = logitsBufferBytes(arch, batchSize)

  if (gpuLayers <= 0 || budgets.length === 0) {
    return {
      fits: true,
      perGpu: budgets.map(() => 0),
      segments: budgets.map(() => ({ weights: 0, kv: 0, compute: 0, cache: 0, extras: 0 })),
      cachePerGpu: budgets.map(() => 0),
      layerSplit: budgets.map(() => 0),
      hostBytes: hostBytes + costs.output + compute + logits,
      kvBytes: kvTotal,
      weightsOnGpu: 0,
      overhead: compute
    }
  }

  items.push(costs.output)
  itemWeights.push(costs.output)
  itemKv.push(0)
  itemStaging.push(0)
  itemCache.push(0)
  weightsOnGpu += costs.output

  // llama.cpp's own figures for configurations like this one, where they have been measured.
  const offset = correctionFor(
    planCorrections,
    { contextLength, kvType, ubatch: batchSize, flashAttention, gpuLayers, cpuMoeLayers: cpuMoe },
    budgets.length
  )
  // Placed against budgets less the correction, so the cut accounts for what the cards will really hold.
  const placed = placeItems(items, budgets.map((b, d) => b - offset[d]), compute, logits)
  const counts = placed.counts
  const perGpu = placed.perGpu.map((need, d) => (counts[d] > 0 ? need + offset[d] : need))
  const segments = deviceSegments(counts, itemWeights, itemKv, itemStaging, compute, logits, itemCache)
  segments.forEach((seg, d) => {
    if (counts[d] > 0) seg.compute = Math.max(0, seg.compute + offset[d])
  })
  const cachePerGpu = segments.map((seg) => seg.cache ?? 0)
  /*
   * No fraction of the card held back.
   *
   * A routed model used to be allowed only 92% of each card, against allocations this engine
   * modelled badly. Those are measured now, and what is kept free is what the user asked for plus
   * an allowance stated in the plan, rather than eight percent of a card nobody chose.
   */
  return {
    fits: perGpu.every((need, i) => need <= budgets[i]),
    perGpu,
    segments,
    cachePerGpu,
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
    predictedVramPerGpu: a.perGpu.map((v, d) => v + (planExtras[d] ?? 0)),
    llamaVramPerGpu: a.perGpu.map((v, d) => v - (a.cachePerGpu[d] ?? 0)),
    cacheVramPerGpu: a.cachePerGpu,
    extrasPerGpu: a.perGpu.map((_, d) => planExtras[d] ?? 0),
    reservePerGpu: a.perGpu.map((_, d) => planKeepFree[d] ?? 0),
    vramSegments: a.segments.map((seg, d) => ({ ...seg, extras: planExtras[d] ?? 0 })),
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
function planFitInner(
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
  const usableGpus = usableDevices(hw)
  const excluded = hw.gpus.filter((g) => !usableGpus.includes(g) && g.totalVram > 0)

  if (excluded.length) {
    notes.push(
      `Excluded from the split: ${excluded.map((g) => g.name).join(', ')} — ` +
        (hw.backend === 'cuda'
          ? 'the CUDA backend can only address NVIDIA devices.'
          : 'integrated GPUs share system memory and would be slower than the discrete cards.')
    )
  }

  /*
   * What each card keeps free, and what it carries outside llama.cpp's own buffers.
   *
   * The reserve is the user's, from Settings, and the safety allowance is stated rather than hidden.
   * The extras (runtime, vision projector, MTP head) are charged to the card that holds them, and
   * counted in that card's prediction, so a load is compared with a figure that includes them. They
   * were once subtracted here and left out of the prediction, so every load looked gigabytes "over"
   * on the first card, and the next plan reserved that again on top.
   */
  const safety = constraints.safetyBytes ?? DEFAULT_SAFETY
  const keepFree = usableGpus.map(() => constraints.headroomBytes + safety)
  const extras =
    constraints.extrasPerGpu && constraints.extrasPerGpu.length === usableGpus.length
      ? constraints.extrasPerGpu
      : priorExtras(usableGpus.length, hw.backend, constraints.companionBytes ?? 0, constraints.draftBytes ?? 0)
  planExtras = extras
  planKeepFree = keepFree
  const budgets = usableGpus.map((g, d) => deviceBudget(g, keepFree[d], extras[d]))
  if (usableGpus.length) {
    notes.push(
      `Keeping ${fmtBytes(constraints.headroomBytes)} free on each card as set in Settings, plus ` +
        `${fmtBytes(safety)} for allocator slack.`
    )
    const extraNote = usableGpus
      .map((g, d) => (extras[d] > 0 ? `${fmtBytes(extras[d])} on ${g.name.replace(/^NVIDIA GeForce /, '')}` : ''))
      .filter(Boolean)
    if (extraNote.length) {
      notes.push(
        `Runtime, vision projector and MTP head: ${extraNote.join(', ')}` +
          (constraints.extrasPerGpu ? ', as measured on the last load.' : ', estimated until the first load measures them.')
      )
    }
  }

  const totalBudget = budgets.reduce((a, b) => a + b, 0)

  const anyUnmeasured = usableGpus.some((g) => !g.freeIsMeasured)
  if (anyUnmeasured) {
    notes.push(
      `Free VRAM could not be measured on ${usableGpus.filter((g) => !g.freeIsMeasured).map((g) => g.name).join(', ')}; ` +
        `assuming ${Math.round(UNMEASURED_FREE_FRACTION * 100)}% of total is available.`
    )
  }
  if (usableGpus.length === 2) {
    notes.push('Layers are split between the two cards wherever that leaves the most room on the tighter one.')
  } else if (usableGpus.length > 2) {
    const pct = proportionalSplit(budgets).map((s) => `${Math.round(s * 100)}%`)
    notes.push(`Split across ${usableGpus.length} GPUs by free capacity: ${pct.join(' / ')}.`)
  }

  // Honour user overrides exactly — they are respected, never silently changed.
  // Buffers are sized by the micro-batch: it is what llama.cpp actually runs through the graph at once.
  const batchSize = resolveBatches(o.batchSize, o.ubatchSize).ubatch
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
    : kvLadder(constraints.preferredKvType, constraints.minKvType, constraints.kvPairs)

  // A pairing chosen by hand is honoured, and explained when this build has to convert it.
  if (o.kvType && constraints.kvPairs && !constraints.kvPairs.includes(kvPairName(kvCandidates[0]))) {
    notes.push(
      `This llama.cpp build has no fast attention kernel for a ${kvLabel(kvCandidates[0])} cache, so it converts ` +
        'the cache to f16 on every generated token: slower, and it needs extra VRAM. Matching key and value types avoids it.'
    )
  }

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

      /*
       * Context stops at the target here, where the other passes grow it to fill the card.
       *
       * What is left over is margin, on purpose. This is the path for a model whose experts are
       * already in system RAM, so the VRAM that more cache would take is the same VRAM that keeps
       * expert layers on the GPU — and the search above has spent as much of it on those as it
       * can. Growing the cache into the rest sized a plan to the last megabyte of an estimate,
       * which is how a 195,584-token plan for Qwen3.8-Flash-Next came to die on a 2 GB cache
       * allocation on a card the planner believed had room. The longer-context trade is offered
       * below instead of taken.
       */
      const ctx = Math.min(
        targetCtx,
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
  const floorKv: KvChoice = kvLadder(constraints.preferredKvType, constraints.minKvType, constraints.kvPairs).at(-1) ?? {
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

/**
 * The logical batch (`--batch-size`) and micro-batch (`--ubatch-size`) a load will use.
 *
 * Either may be left unset (or 0) for automatic, which is 512 — what this app has always passed.
 * llama.cpp never runs a micro-batch larger than the batch, so an explicit micro-batch above an
 * automatic batch raises the batch to match rather than being quietly cut back to 512.
 */
export function resolveBatches(batch?: number, ubatch?: number): { batch: number; ubatch: number } {
  const b = batch && batch > 0 ? Math.round(batch) : 0
  const ub = ubatch && ubatch > 0 ? Math.round(ubatch) : 0
  if (b && ub) return { batch: b, ubatch: Math.min(ub, b) }
  if (b) return { batch: b, ubatch: Math.min(512, b) }
  if (ub) return { batch: Math.max(512, ub), ubatch: ub }
  return { batch: 512, ubatch: 512 }
}

function withCacheSlots(arch: ModelArchInfo, slots: number, plan: () => FitResult): FitResult {
  planExpertCacheSlots = Math.max(0, Math.min(slots, Math.max(0, arch.expertCount - 1)))
  try {
    const result = plan()
    const chosen = result.chosen
    if (chosen && planExpertCacheSlots > 0 && (chosen.cpuMoeLayers ?? 0) > 0) {
      const costs = layerCosts(arch)
      let bytes = 0
      for (let il = 0; il < (chosen.cpuMoeLayers ?? 0); il++) bytes += expertCacheBytes(arch, costs.experts[il])
      result.chosen = {
        ...chosen,
        expertCacheSlots: planExpertCacheSlots,
        rationale: [
          ...chosen.rationale,
          `GPU expert cache: the ${planExpertCacheSlots} most recently used of ${arch.expertCount} experts per layer are kept on the GPUs (${fmtBytes(bytes)}) and computed there.`
        ]
      }
    }
    return result
  } finally {
    planExpertCacheSlots = 0
  }
}

/*
 * The GPU expert cache, sized to what the cards have left.
 *
 * When a mixture-of-experts model spills experts to system RAM, VRAM spent on cache slots buys far
 * more speed than the same VRAM spent on whole expert layers: on Qwen3.8-Flash-Next 16 GB of cache
 * took generation from 18 to 41 tokens/s. So every expert stays in RAM, the context the plain plan
 * chose is kept, and the cache takes the rest. The slot count is searched rather than computed
 * because each layer's share lands on the card that holds the layer, so the cards fill unevenly.
 */
const EXPERT_CACHE_MIN_SLOTS = 16
/*
 * Slots held back from what the search found. Two, where it was eight: the plans are measured now,
 * and what guards against allocator slack is the stated allowance every card keeps free.
 */
const EXPERT_CACHE_MARGIN_SLOTS = 2

function autoExpertCache(arch: ModelArchInfo, hw: HardwareSnapshot, constraints: FitConstraints): FitResult {
  const plain = withCacheSlots(arch, 0, () => planFitInner(arch, hw, constraints))
  const chosen = plain.chosen
  if (!chosen || arch.expertCount <= 0 || (chosen.cpuMoeLayers ?? 0) <= 0) return plain

  const o = constraints.overrides
  const pinned: FitConstraints = {
    ...constraints,
    overrides: {
      ...o,
      cpuMoeLayers: o.cpuMoeLayers ?? arch.blockCount,
      contextLength: chosen.contextLength,
      kvType: chosen.kvType,
      kvTypeV: chosen.kvTypeV
    }
  }
  const keeps = (r: FitResult): boolean =>
    !!r.chosen && r.chosen.contextLength === chosen.contextLength && r.chosen.kvType === chosen.kvType
  const at = (slots: number): FitResult => withCacheSlots(arch, slots, () => planFitInner(arch, hw, pinned))

  if (!keeps(at(EXPERT_CACHE_MIN_SLOTS))) return plain
  let lo = EXPERT_CACHE_MIN_SLOTS
  let hi = arch.expertCount - 1
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2)
    if (keeps(at(mid))) lo = mid
    else hi = mid - 1
  }
  // A margin against allocator slack, unless every expert already fits.
  const slots = lo >= arch.expertCount - 1 ? lo : Math.max(EXPERT_CACHE_MIN_SLOTS, lo - EXPERT_CACHE_MARGIN_SLOTS)
  const result = at(slots)

  /*
   * Offer the faster cache, without taking it.
   *
   * With every expert in system RAM, VRAM the cache does not get is speed lost, and the value half
   * of the KV cache is the one that tolerates four bits (about one changed answer in five hundred).
   * So when coarser values would buy a real number of extra slots, that plan is put beside the chosen
   * one rather than picked: the user's cache preference stands unless they choose otherwise.
   */
  const coarser: KvChoice = { k: chosen.kvType, v: 'q4_0' }
  const allowed = !o.kvType && chosen.kvTypeV !== 'q4_0' && (!constraints.kvPairs || constraints.kvPairs.includes(kvPairName(coarser)))
  if (allowed && result.chosen) {
    const fast: FitConstraints = { ...pinned, overrides: { ...pinned.overrides, kvType: coarser.k, kvTypeV: coarser.v } }
    const fastAt = (n: number): FitResult => withCacheSlots(arch, n, () => planFitInner(arch, hw, fast))
    const fastKeeps = (r: FitResult): boolean => !!r.chosen && r.chosen.contextLength === chosen.contextLength
    let flo = slots
    let fhi = arch.expertCount - 1
    if (fastKeeps(fastAt(flo))) {
      while (flo < fhi) {
        const mid = Math.ceil((flo + fhi) / 2)
        if (fastKeeps(fastAt(mid))) flo = mid
        else fhi = mid - 1
      }
      const fastSlots = flo >= arch.expertCount - 1 ? flo : Math.max(slots, flo - EXPERT_CACHE_MARGIN_SLOTS)
      const fastPlan = fastAt(fastSlots).chosen
      // Worth a button only if it is a step, not a rounding difference.
      if (fastPlan && fastSlots >= slots + Math.max(8, slots * 0.25)) {
        result.alternatives = [
          {
            ...fastPlan,
            label: 'Faster',
            rationale: [
              ...fastPlan.rationale,
              `Values at q4_0 instead of ${chosen.kvTypeV} free room for ${fastSlots - slots} more expert-cache slots ` +
                `(${fastSlots} against ${slots}), so more of each token's experts are read from VRAM rather than system RAM.`,
              'Keys are unchanged, which is the half that carries the accuracy; coarser values measure at roughly one changed answer in five hundred.'
            ]
          },
          ...result.alternatives
        ]
      }
    }
  }
  return result
}

export function planFit(
  arch: ModelArchInfo,
  hw: HardwareSnapshot,
  constraints: FitConstraints = DEFAULT_CONSTRAINTS
): FitResult {
  const requested = Math.floor(constraints.expertCacheSlots ?? 0)
  planCorrections = constraints.corrections ?? []
  let result: FitResult
  try {
    result =
      requested < 0
        ? autoExpertCache(arch, hw, constraints)
        : withCacheSlots(arch, requested, () => planFitInner(arch, hw, constraints))
  } finally {
    planCorrections = []
    planExtras = []
    planKeepFree = []
  }
  const { batch, ubatch } = resolveBatches(constraints.overrides.batchSize, constraints.overrides.ubatchSize)
  /*
   * Name the cards each plan's per-card figures belong to.
   *
   * The figures follow the devices the backend can address, which is not always every adapter in
   * the snapshot — an integrated GPU beside discrete cards is left out — so they cannot be paired
   * with `hardware.gpus` by position.
   */
  const hasDiscrete = hw.gpus.some((g) => g.totalVram > 0 && !/\bgraphics\b/i.test(g.name))
  const devices: PlanDevice[] = hw.gpus
    .filter((g) => usableForBackend(g, hw.backend, hasDiscrete))
    .map((g) => ({ index: g.index, name: g.name, totalVram: g.totalVram, freeVram: g.freeVram, measured: g.freeIsMeasured }))
  return {
    ...result,
    chosen: result.chosen ? { ...result.chosen, devices, batchSize: batch, ubatchSize: ubatch } : null,
    alternatives: result.alternatives.map((p) => ({ ...p, devices, batchSize: batch, ubatchSize: ubatch }))
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
