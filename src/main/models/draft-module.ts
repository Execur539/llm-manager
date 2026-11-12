/**
 * A multi-token-prediction head shipped beside a model, and how it is run.
 *
 * Unsloth ships Qwen3.8-Flash-Next's MTP head as its own file (`MTP/mtp-*.gguf`): the one trailing
 * block plus a copy of the token embedding and LM head, and none of the trunk. llama.cpp loads it
 * as the draft model of `--spec-type draft-mtp`. Pure, so the smoke suite can hold it to the real
 * file's layout.
 *
 * Placement was measured on Flash-Next with the GPU expert cache (RTX 5080 + 4070 Ti, equal VRAM):
 * the head on the GPU whole cost 48 cache slots and ran slower than no head at all, while keeping
 * its 512 experts and its embedding table in system RAM left ~0.75 GB of weights on the GPU and was
 * fastest of everything tried. With the app's own plan (214K context, ubatch 2048: 105 cache slots
 * without the head, 85 with it) it drafted 2 tokens best: +9% on prose, +40% rewriting code, +4%
 * on sampled prose; 3 tokens gained more on code but nothing on prose.
 */

/** Tokens a separate head drafts at most: at 3 its prose acceptance fell to 48% and gained nothing. */
export const SIDECAR_MAX_DRAFT = 2

/** What matching and reserving need from a head file's header. */
export interface DraftModuleInfo {
  path: string
  architecture: string
  embeddingLength: number
  vocabSize: number
  /** Blocks of the model it belongs to: the file's block count less its MTP blocks. */
  trunkBlocks: number
  /** Bytes it puts on the GPU with its experts and embedding table kept in system RAM. */
  gpuBytes: number
}

/** What a matching model has to agree on. */
export interface DraftModuleTarget {
  path: string
  architecture: string
  embeddingLength: number
  vocabSize: number
  blockCount: number
  /** MTP blocks inside the model's own file; a model carrying its own head needs no other. */
  mtpLayers: number
}

/** Tensors that stay in system RAM: the routed experts and the token embedding table. */
const HOST_TENSOR_RE = /(^|\.)ffn_(gate|up|down|gate_up)_exps\.|^token_embd\./

interface HeaderLike {
  kv: Record<string, unknown>
  tensors: { name: string; dims: number[]; bytes: number }[]
}

/** A head file's header, or null when it is not one (no MTP blocks, or no trunk to belong to). */
export function draftModuleInfo(path: string, meta: HeaderLike): DraftModuleInfo | null {
  const architecture = meta.kv['general.architecture']
  if (typeof architecture !== 'string' || !architecture) return null
  const key = (name: string): number => Number(meta.kv[`${architecture}.${name}`] ?? 0) || 0
  const nextn = key('nextn_predict_layers')
  const blocks = key('block_count')
  if (nextn <= 0 || blocks <= nextn) return null
  const embd = meta.tensors.find((t) => t.name === 'token_embd.weight')
  const gpuBytes = meta.tensors.filter((t) => !HOST_TENSOR_RE.test(t.name)).reduce((sum, t) => sum + t.bytes, 0)
  return {
    path,
    architecture,
    embeddingLength: key('embedding_length'),
    vocabSize: embd?.dims[1] ?? 0,
    trunkBlocks: blocks - nextn,
    gpuBytes
  }
}

function dirOf(p: string): string {
  const parts = p.replace(/\\/g, '/').split('/')
  parts.pop()
  return parts.join('/').toLowerCase()
}

/**
 * The head that belongs to a model, nearest first: the model's own folder, then an `MTP` folder
 * beside it, then any head in the library built for the same trunk (the same architecture, width,
 * vocabulary and depth) — which is what lets a re-quantised or edited copy of a model elsewhere
 * use the head that came with the original.
 */
export function pickDraftModule(model: DraftModuleTarget, heads: readonly DraftModuleInfo[]): DraftModuleInfo | null {
  if (model.mtpLayers > 0) return null
  const fits = heads.filter(
    (h) =>
      h.architecture === model.architecture &&
      h.embeddingLength === model.embeddingLength &&
      h.vocabSize === model.vocabSize &&
      h.trunkBlocks === model.blockCount
  )
  if (!fits.length) return null
  const home = dirOf(model.path)
  const rank = (h: DraftModuleInfo): number => {
    const dir = dirOf(h.path)
    if (dir === home) return 0
    if (dir === `${home}/mtp`) return 1
    return 2
  }
  return [...fits].sort((a, b) => rank(a) - rank(b))[0]
}

/**
 * The draft context's GPU compute buffer, which grows with the batch size it inherits from the model.
 *
 * Measured on Flash-Next's head (RTX 5080): 742 MiB at ubatch 512, 1,064 at 1024, 1,709 at 2048 --
 * exactly 419 MiB + 0.63 MiB a token. A flat 512 MiB was assumed at first, and at the app's
 * 2048-token batch the head then failed to start for want of 1.2 GB; the DLL now caps the draft
 * context's ubatch (MTP_MAX_UBATCH). Charged with 5% to spare.
 */
export function draftComputeBytes(ubatch: number): number {
  const MiB = 1024 * 1024
  return Math.ceil((419.3 * MiB + 0.63 * MiB * Math.max(1, ubatch)) * 1.05)
}

/**
 * VRAM the head takes on the first GPU: its weights there, its one layer of cache for the whole
 * context, and its compute buffer. Keys and values at q8_0 size, the type the planner prefers.
 */
export function draftReserveBytes(
  head: Pick<DraftModuleInfo, 'gpuBytes'>,
  arch: { headCountKv: number; headDim: number },
  contextLength: number,
  ubatch: number
): number {
  const kv = 2 * arch.headCountKv * arch.headDim * contextLength * (34 / 32)
  return head.gpuBytes + Math.ceil(kv) + draftComputeBytes(Math.min(ubatch, MTP_MAX_UBATCH))
}

/**
 * The bundled llama.dll caps an MTP draft context's ubatch at this (qwen4exp-mtp.patch,
 * llama-context.cpp): it decodes a few tokens at a time, and its compute buffer grew with the
 * target's ubatch for nothing.
 */
export const MTP_MAX_UBATCH = 512

/** llama-server arguments that attach the head, placed as measured best. */
export function draftModuleArgs(path: string, kvType: string, kvTypeV: string): string[] {
  return [
    '--spec-draft-model', path,
    '--spec-draft-ngl', '999',
    // The primary device, where the planner charged it.
    '--spec-draft-device', 'CUDA0',
    '--spec-draft-cpu-moe',
    '--spec-draft-override-tensor', 'token_embd\\.weight=CPU',
    '--spec-draft-type-k', kvType,
    '--spec-draft-type-v', kvTypeV
  ]
}
