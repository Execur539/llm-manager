/**
 * HuggingFace search and quant recommendation.
 *
 * No model list is hardcoded anywhere — a model released this morning is findable this
 * afternoon. Recommendation is computed from the user's actual free VRAM via the same
 * auto-fit maths used at load time, so the badge in search agrees with what happens later.
 */

import type { GgufValue, HardwareSnapshot, ModelArchInfo } from '@shared/types'

/**
 * HuggingFace API root.
 *
 * Overridable so the download flow can be exercised end to end without network access, and so a
 * user behind a mirror can point at it. Only the API base moves; resolved file URLs come from
 * the API response itself.
 */
const HF_BASE = process.env.LLMM_HF_BASE ?? 'https://huggingface.co'
const HF_API = `${HF_BASE}/api`

export interface HfModelSummary {
  id: string
  downloads: number
  likes: number
  updatedAt: string
  tags: string[]
  gated: boolean
}

export interface HfFile {
  filename: string
  bytes: number
  /** parsed from the filename, e.g. "Q4_K_M" */
  quant: string | null
  url: string
  isMmproj: boolean
  /** part 1 of N for a split model, when applicable */
  shard: { index: number; total: number } | null
}

export interface QuantRecommendation {
  filename: string
  reason: string
  /** predicted context if loaded now */
  predictedContext: number
  fitsFullyOnGpu: boolean
}

function authHeaders(token: string | null): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export async function searchModels(
  query: string,
  token: string | null,
  limit = 25
): Promise<HfModelSummary[]> {
  // `gguf` filter keeps results to repos that actually contain something we can run.
  const url = `${HF_API}/models?search=${encodeURIComponent(query)}&filter=gguf&sort=downloads&direction=-1&limit=${limit}&full=false`
  const res = await fetch(url, { headers: authHeaders(token) })
  if (res.status === 401) throw new Error('HuggingFace rejected the token. Check it in Settings.')
  if (!res.ok) throw new Error(`HuggingFace search failed: HTTP ${res.status}`)

  const json = (await res.json()) as {
    id: string
    downloads?: number
    likes?: number
    lastModified?: string
    tags?: string[]
    gated?: boolean | string
  }[]

  return json.map((m) => ({
    id: m.id,
    downloads: m.downloads ?? 0,
    likes: m.likes ?? 0,
    updatedAt: m.lastModified ?? '',
    tags: m.tags ?? [],
    gated: Boolean(m.gated)
  }))
}

const QUANT_RE = /\b(IQ\d[A-Z_]*|Q\d(?:_[A-Z0-9]+)*|F16|BF16|F32)\b/i

export async function listFiles(repo: string, token: string | null): Promise<HfFile[]> {
  const res = await fetch(`${HF_API}/models/${repo}/tree/main?recursive=true`, {
    headers: authHeaders(token)
  })
  if (res.status === 403) {
    throw new Error(`${repo} is gated. Accept its licence on huggingface.co and add a token in Settings.`)
  }
  if (!res.ok) throw new Error(`Could not list ${repo}: HTTP ${res.status}`)

  const tree = (await res.json()) as { path: string; size?: number; type: string; lfs?: { size: number } }[]

  return tree
    .filter((f) => f.type === 'file' && f.path.toLowerCase().endsWith('.gguf'))
    .map((f) => {
      const filename = f.path
      const base = filename.split('/').pop() ?? filename
      const shardMatch = base.match(/-(\d{5})-of-(\d{5})\.gguf$/i)
      return {
        filename,
        bytes: f.lfs?.size ?? f.size ?? 0,
        quant: base.match(QUANT_RE)?.[0]?.toUpperCase() ?? null,
        url: `${HF_BASE}/${repo}/resolve/main/${encodeURI(filename)}`,
        isMmproj: /mmproj/i.test(base),
        shard: shardMatch ? { index: Number(shardMatch[1]), total: Number(shardMatch[2]) } : null
      }
    })
    .sort((a, b) => a.bytes - b.bytes)
}

/**
 * Recommend a quant for this machine.
 *
 * We do not have the GGUF header before downloading, so weights are taken from the file size
 * and the KV cache is estimated from a typical modern GQA geometry. It is approximate by
 * necessity — the exact fit is recomputed from real metadata once the file is on disk.
 */
export function recommendQuant(
  files: HfFile[],
  hardware: HardwareSnapshot,
  targetContext: number
): QuantRecommendation | null {
  const candidates = files.filter((f) => !f.isMmproj && (!f.shard || f.shard.index === 1))
  if (!candidates.length) return null

  const freeVram = hardware.gpus.reduce(
    (sum, g) => sum + (g.freeIsMeasured && g.freeVram >= 0 ? g.freeVram : g.totalVram * 0.85),
    0
  )
  // Headroom for the compute buffer, CUDA context and desktop use.
  const usable = Math.max(0, freeVram - 1.5 * 1024 ** 3)

  // KV estimate: 8 KV heads x 128 dim x 2 x ~48 layers at q8_0 ≈ 105 KB per 1k tokens.
  const kvPerToken = 2 * 8 * 128 * 48 * (34 / 32)
  const kvForTarget = kvPerToken * targetContext

  const ranked = candidates
    .map((f) => {
      const totalNeed = f.bytes + kvForTarget
      const fits = totalNeed <= usable
      const spare = usable - f.bytes
      const predictedContext = spare > 0 ? Math.floor(spare / kvPerToken / 1024) * 1024 : 0
      return { file: f, fits, predictedContext, quality: qualityRank(f.quant) }
    })
    // Best quality that still fits, then largest context.
    .sort((a, b) => {
      if (a.fits !== b.fits) return a.fits ? -1 : 1
      if (a.fits && b.fits) return b.quality - a.quality
      return b.predictedContext - a.predictedContext
    })

  const best = ranked[0]
  if (!best) return null

  const gb = (n: number): string => `${(n / 1024 ** 3).toFixed(1)} GB`

  return {
    filename: best.file.filename,
    fitsFullyOnGpu: best.fits,
    predictedContext: Math.min(best.predictedContext, 1_048_576),
    reason: best.fits
      ? `${best.file.quant ?? 'this file'} is ${gb(best.file.bytes)} and fits your ${gb(freeVram)} of free VRAM with room for roughly ${best.predictedContext.toLocaleString()} tokens of context.`
      : `Nothing here fits fully in your ${gb(freeVram)} of free VRAM. ${best.file.quant ?? 'This file'} at ${gb(best.file.bytes)} is the closest — expect partial CPU offload.`
  }
}

/** Higher is better quality. Ordering follows the usual llama.cpp quant hierarchy. */
function qualityRank(quant: string | null): number {
  if (!quant) return 0
  const q = quant.toUpperCase()
  const table: Record<string, number> = {
    F32: 100, BF16: 95, F16: 95,
    Q8_0: 90,
    Q6_K: 80,
    Q5_K_M: 74, Q5_K_S: 72, Q5_1: 70, Q5_0: 69,
    Q4_K_M: 64, Q4_K_S: 62, Q4_1: 60, Q4_0: 58,
    IQ4_XS: 56, IQ4_NL: 55,
    Q3_K_L: 48, Q3_K_M: 46, Q3_K_S: 44,
    IQ3_M: 42, IQ3_S: 40, IQ3_XXS: 38,
    Q2_K: 30, IQ2_M: 26, IQ2_S: 24, IQ2_XXS: 22,
    IQ1_M: 12, IQ1_S: 10
  }
  return table[q] ?? 50
}

/** Find the mmproj companion for a chosen model file, when the repo has one. */
export function findMmprojFor(files: HfFile[]): HfFile | null {
  const projectors = files.filter((f) => f.isMmproj)
  if (!projectors.length) return null
  // Prefer an f16 projector; they are small and highest quality.
  return projectors.find((f) => /f16/i.test(f.filename)) ?? projectors[0]
}

/** Metadata peek: read just the GGUF header over HTTP range requests, before downloading. */
export async function peekRemoteGguf(
  url: string,
  token: string | null
): Promise<{ kv: Record<string, GgufValue>; partial: true } | null> {
  try {
    const res = await fetch(url, {
      headers: { ...authHeaders(token), Range: 'bytes=0-1048575' }
    })
    if (!res.ok && res.status !== 206) return null
    // Full parsing needs the tensor directory too; this only confirms the magic and version,
    // which is enough to catch a mislabelled file before spending 20 GB of bandwidth.
    const buf = Buffer.from(await res.arrayBuffer())
    if (buf.length < 8) return null
    const magic = buf.readUInt32LE(0)
    if (magic !== 0x46554747) return null
    return { kv: { 'gguf.version': buf.readUInt32LE(4) }, partial: true }
  } catch {
    return null
  }
}

export function estimateArchFromSize(bytes: number): Partial<ModelArchInfo> {
  // Only used for display before download; real values come from the header afterwards.
  return { weightBytes: bytes }
}
