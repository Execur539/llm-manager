/**
 * GGUF metadata reader.
 *
 * Parses the header, metadata key/value block and tensor directory of a .gguf file
 * without loading tensor data. Large arrays (notably `tokenizer.ggml.tokens`, which can
 * hold 100k+ strings) are walked but not retained, so parsing a 40 GB model reads only
 * a few megabytes.
 *
 * Format reference: magic "GGUF", u32 version, u64 tensor_count, u64 kv_count,
 * then kv pairs, then tensor infos, then padding to `general.alignment`, then data.
 */

import { open, stat, FileHandle } from 'node:fs/promises'
import type { GgufMetadata, GgufTensorInfo, GgufValue, ModelArchInfo } from '@shared/types'

const GGUF_MAGIC = 0x46554747 // "GGUF" little-endian

enum GgufType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12
}

/** Keys whose array values are huge and never needed; walked and discarded. */
const ELIDE_ARRAY_KEYS = new Set([
  'tokenizer.ggml.tokens',
  'tokenizer.ggml.scores',
  'tokenizer.ggml.token_type',
  'tokenizer.ggml.merges'
])

/** [blockSize, bytesPerBlock] for each ggml tensor type. */
const GGML_TYPE_TRAITS: Record<number, [number, number]> = {
  0: [1, 4], // F32
  1: [1, 2], // F16
  2: [32, 18], // Q4_0
  3: [32, 20], // Q4_1
  6: [32, 22], // Q5_0
  7: [32, 24], // Q5_1
  8: [32, 34], // Q8_0
  9: [32, 36], // Q8_1
  10: [256, 84], // Q2_K
  11: [256, 110], // Q3_K
  12: [256, 144], // Q4_K
  13: [256, 176], // Q5_K
  14: [256, 210], // Q6_K
  15: [256, 292], // Q8_K
  16: [256, 66], // IQ2_XXS
  17: [256, 74], // IQ2_XS
  18: [256, 98], // IQ3_XXS
  19: [256, 50], // IQ1_S
  20: [32, 18], // IQ4_NL
  21: [256, 110], // IQ3_S
  22: [256, 82], // IQ2_S
  23: [256, 136], // IQ4_XS
  24: [1, 1], // I8
  25: [1, 2], // I16
  26: [1, 4], // I32
  27: [1, 8], // I64
  28: [1, 8], // F64
  29: [256, 56], // IQ1_M
  30: [1, 2] // BF16
}

const GGML_TYPE_NAMES: Record<number, string> = {
  0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1',
  10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS',
  17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S',
  23: 'IQ4_XS', 24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64', 29: 'IQ1_M', 30: 'BF16'
}

export function tensorByteSize(ggmlType: number, dims: number[]): number {
  const traits = GGML_TYPE_TRAITS[ggmlType]
  const elements = dims.reduce((a, b) => a * b, 1)
  if (!traits) return elements // unknown type: assume 1 byte/element rather than throwing
  const [blockSize, bytesPerBlock] = traits
  return Math.ceil(elements / blockSize) * bytesPerBlock
}

/**
 * Sliding-window reader over a file handle. Keeps a buffer and refills it as the parser
 * advances, so we never need the whole file (or even the whole metadata block) in memory.
 */
class ChunkReader {
  private buf = Buffer.alloc(0)
  /** absolute file offset of buf[0] */
  private bufStart = 0
  /** absolute file offset of the cursor */
  private pos = 0

  constructor(
    private fh: FileHandle,
    private fileSize: number,
    private chunkSize = 1 << 20
  ) {}

  get offset(): number {
    return this.pos
  }

  private get local(): number {
    return this.pos - this.bufStart
  }

  /** Guarantee `n` bytes are available at the cursor. */
  private async ensure(n: number): Promise<void> {
    if (this.local >= 0 && this.local + n <= this.buf.length) return
    if (this.pos + n > this.fileSize) {
      throw new Error(`GGUF: unexpected end of file at ${this.pos} (needed ${n} bytes)`)
    }
    // Discard everything before the cursor and read forward.
    const want = Math.max(n, this.chunkSize)
    const size = Math.min(want, this.fileSize - this.pos)
    const next = Buffer.alloc(size)
    await this.fh.read(next, 0, size, this.pos)
    this.buf = next
    this.bufStart = this.pos
  }

  async u8(): Promise<number> {
    await this.ensure(1)
    const v = this.buf.readUInt8(this.local)
    this.pos += 1
    return v
  }
  async i8(): Promise<number> {
    await this.ensure(1)
    const v = this.buf.readInt8(this.local)
    this.pos += 1
    return v
  }
  async u16(): Promise<number> {
    await this.ensure(2)
    const v = this.buf.readUInt16LE(this.local)
    this.pos += 2
    return v
  }
  async i16(): Promise<number> {
    await this.ensure(2)
    const v = this.buf.readInt16LE(this.local)
    this.pos += 2
    return v
  }
  async u32(): Promise<number> {
    await this.ensure(4)
    const v = this.buf.readUInt32LE(this.local)
    this.pos += 4
    return v
  }
  async i32(): Promise<number> {
    await this.ensure(4)
    const v = this.buf.readInt32LE(this.local)
    this.pos += 4
    return v
  }
  async f32(): Promise<number> {
    await this.ensure(4)
    const v = this.buf.readFloatLE(this.local)
    this.pos += 4
    return v
  }
  async f64(): Promise<number> {
    await this.ensure(8)
    const v = this.buf.readDoubleLE(this.local)
    this.pos += 8
    return v
  }
  async u64(): Promise<number> {
    await this.ensure(8)
    const v = this.buf.readBigUInt64LE(this.local)
    this.pos += 8
    // Counts and offsets in practice fit comfortably in a double.
    return Number(v)
  }
  async i64(): Promise<number> {
    await this.ensure(8)
    const v = this.buf.readBigInt64LE(this.local)
    this.pos += 8
    return Number(v)
  }
  async bool(): Promise<boolean> {
    return (await this.u8()) !== 0
  }
  async str(): Promise<string> {
    const len = await this.u64()
    if (len > 64 * 1024 * 1024) throw new Error(`GGUF: implausible string length ${len}`)
    await this.ensure(len)
    const s = this.buf.toString('utf8', this.local, this.local + len)
    this.pos += len
    return s
  }
  /** Skip a string without materialising it. */
  async skipStr(): Promise<void> {
    const len = await this.u64()
    this.pos += len
    if (this.pos > this.fileSize) throw new Error('GGUF: string overruns file')
  }
  skip(n: number): void {
    this.pos += n
  }
}

const FIXED_WIDTHS: Partial<Record<GgufType, number>> = {
  [GgufType.UINT8]: 1,
  [GgufType.INT8]: 1,
  [GgufType.BOOL]: 1,
  [GgufType.UINT16]: 2,
  [GgufType.INT16]: 2,
  [GgufType.UINT32]: 4,
  [GgufType.INT32]: 4,
  [GgufType.FLOAT32]: 4,
  [GgufType.UINT64]: 8,
  [GgufType.INT64]: 8,
  [GgufType.FLOAT64]: 8
}

async function readScalar(r: ChunkReader, type: GgufType): Promise<GgufValue> {
  switch (type) {
    case GgufType.UINT8: return r.u8()
    case GgufType.INT8: return r.i8()
    case GgufType.UINT16: return r.u16()
    case GgufType.INT16: return r.i16()
    case GgufType.UINT32: return r.u32()
    case GgufType.INT32: return r.i32()
    case GgufType.FLOAT32: return r.f32()
    case GgufType.BOOL: return r.bool()
    case GgufType.STRING: return r.str()
    case GgufType.UINT64: return r.u64()
    case GgufType.INT64: return r.i64()
    case GgufType.FLOAT64: return r.f64()
    default:
      throw new Error(`GGUF: unsupported value type ${type}`)
  }
}

/** Walk past an array without retaining its contents. */
async function skipArray(r: ChunkReader, elemType: GgufType, count: number): Promise<void> {
  const width = FIXED_WIDTHS[elemType]
  if (width !== undefined) {
    r.skip(width * count)
    return
  }
  if (elemType === GgufType.STRING) {
    for (let i = 0; i < count; i++) await r.skipStr()
    return
  }
  throw new Error(`GGUF: cannot skip nested array of type ${elemType}`)
}

export async function readGguf(path: string): Promise<GgufMetadata> {
  const info = await stat(path)
  const fh = await open(path, 'r')
  try {
    const r = new ChunkReader(fh, info.size)

    const magic = await r.u32()
    if (magic !== GGUF_MAGIC) {
      throw new Error(`Not a GGUF file (magic 0x${magic.toString(16)})`)
    }
    const version = await r.u32()
    if (version < 2 || version > 3) {
      // v1 used u32 counts; refuse rather than silently misparse.
      throw new Error(`Unsupported GGUF version ${version}`)
    }
    const tensorCount = await r.u64()
    const kvCount = await r.u64()

    const kv: Record<string, GgufValue> = {}
    for (let i = 0; i < kvCount; i++) {
      const key = await r.str()
      const type = (await r.u32()) as GgufType
      if (type === GgufType.ARRAY) {
        const elemType = (await r.u32()) as GgufType
        const count = await r.u64()
        if (ELIDE_ARRAY_KEYS.has(key) || count > 4096) {
          await skipArray(r, elemType, count)
          kv[key] = { elided: true, type: elemType, count }
        } else {
          const arr: GgufValue[] = []
          for (let j = 0; j < count; j++) arr.push(await readScalar(r, elemType))
          kv[key] = arr
        }
      } else {
        kv[key] = await readScalar(r, type)
      }
    }

    const tensors: GgufTensorInfo[] = []
    for (let i = 0; i < tensorCount; i++) {
      const name = await r.str()
      const nDims = await r.u32()
      const dims: number[] = []
      for (let d = 0; d < nDims; d++) dims.push(await r.u64())
      const ggmlType = await r.u32()
      const offset = await r.u64()
      tensors.push({ name, dims, ggmlType, offset, bytes: tensorByteSize(ggmlType, dims) })
    }

    const alignment = typeof kv['general.alignment'] === 'number' ? (kv['general.alignment'] as number) : 32
    const dataOffset = Math.ceil(r.offset / alignment) * alignment

    return { version, tensorCount, kv, tensors, dataOffset }
  } finally {
    await fh.close()
  }
}

function num(kv: Record<string, GgufValue>, key: string): number | null {
  const v = kv[key]
  return typeof v === 'number' ? v : null
}

/**
 * Distil the raw metadata into the facts the auto-fit engine needs.
 *
 * Layer/non-layer weight split matters: only tensors inside repeating blocks can be
 * offloaded per-layer, so budgeting has to treat them separately from embeddings and
 * the output head.
 */
export function extractArchInfo(meta: GgufMetadata): ModelArchInfo {
  const kv = meta.kv
  const arch = typeof kv['general.architecture'] === 'string' ? (kv['general.architecture'] as string) : 'unknown'
  const p = (suffix: string) => num(kv, `${arch}.${suffix}`)

  const blockCount = p('block_count') ?? 0
  const embeddingLength = p('embedding_length') ?? 0
  const headCount = p('attention.head_count') ?? 0
  const headCountKv = p('attention.head_count_kv') ?? headCount
  const contextLength = p('context_length') ?? 0
  const expertCount = p('expert_count') ?? 0

  // head_dim is stated explicitly by newer converters; otherwise derive it.
  let headDim = p('attention.head_dim') ?? p('attention.key_length') ?? 0
  if (!headDim && headCount > 0 && embeddingLength > 0) {
    headDim = Math.floor(embeddingLength / headCount)
  }

  const tokensKv = kv['tokenizer.ggml.tokens']
  const vocabSize =
    tokensKv && typeof tokensKv === 'object' && 'elided' in tokensKv
      ? tokensKv.count
      : Array.isArray(tokensKv)
        ? tokensKv.length
        : (p('vocab_size') ?? 0)

  // Weight accounting. Block tensors are named "blk.<n>." by convention across architectures.
  let perLayerBytes = 0
  let nonLayerBytes = 0
  const quantCounts = new Map<number, number>()
  for (const t of meta.tensors) {
    if (/^blk\.\d+\./.test(t.name)) perLayerBytes += t.bytes
    else nonLayerBytes += t.bytes
    // Only count 2D+ weight tensors toward "the" quant; 1D norms are always F32.
    if (t.dims.length >= 2) {
      quantCounts.set(t.ggmlType, (quantCounts.get(t.ggmlType) ?? 0) + t.bytes)
    }
  }

  let dominant = 0
  let best = -1
  for (const [type, bytes] of quantCounts) {
    if (bytes > best) {
      best = bytes
      dominant = type
    }
  }

  // Hybrid attention/SSM layout. `full_attention_interval = 4` means one in every four layers
  // is full attention and the rest are state-space layers, whose state is a fixed size rather
  // than one that grows with context.
  const fullAttentionInterval = p('full_attention_interval') ?? 0
  const ssmInnerSize = p('ssm.inner_size') ?? 0
  const ssmStateSize = p('ssm.state_size') ?? 0
  const ssmConvKernel = p('ssm.conv_kernel') ?? 0
  const isHybrid = fullAttentionInterval > 1 && ssmStateSize > 0

  const attentionLayers = isHybrid ? Math.ceil(blockCount / fullAttentionInterval) : blockCount
  const ssmLayers = isHybrid ? blockCount - attentionLayers : 0

  // Recurrent state per SSM layer: the convolution window plus the recurrent state itself.
  // llama.cpp keeps both in f32.
  const ssmStateBytesPerLayer = isHybrid
    ? (ssmInnerSize * Math.max(0, ssmConvKernel - 1) + ssmInnerSize * ssmStateSize) * 4
    : 0

  return {
    architecture: arch,
    name: typeof kv['general.name'] === 'string' ? (kv['general.name'] as string) : null,
    blockCount,
    embeddingLength,
    headCount,
    headCountKv,
    headDim,
    contextLength,
    vocabSize,
    quant: GGML_TYPE_NAMES[dominant] ?? `type${dominant}`,
    weightBytes: perLayerBytes + nonLayerBytes,
    perLayerBytes,
    nonLayerBytes,
    expertCount,
    attentionLayers,
    ssmLayers,
    ssmStateBytesPerLayer
  }
}

/**
 * Detect what a chat template implies about tool support.
 * llama.cpp can force valid tool calls via GBNF regardless, but a template that already
 * knows about tools produces much better results, so we surface the difference.
 */
export function templateSupportsTools(meta: GgufMetadata): boolean {
  const tpl = meta.kv['tokenizer.chat_template']
  if (typeof tpl !== 'string') return false
  return /tool|function_call|tool_calls/i.test(tpl)
}
