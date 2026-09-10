/**
 * HuggingFace search and quant recommendation.
 *
 * No model list is hardcoded anywhere — a model released this morning is findable this
 * afternoon. Recommendation is computed from the user's actual free VRAM via the same
 * auto-fit maths used at load time, so the badge in search agrees with what happens later.
 */

import path from 'node:path'
import os from 'node:os'
import fsp from 'node:fs/promises'
import crypto from 'node:crypto'
import type { GgufValue, HardwareSnapshot, ModelArchInfo } from '@shared/types'
import { readGguf, extractArchInfo } from '../models/gguf'

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
  /**
   * SHA-256 of the file contents, when HuggingFace publishes one.
   *
   * From `lfs.oid`, not the sibling `oid` — that one is the git blob SHA-1 and would never match.
   * Absent for anything not stored in LFS, which in practice means anything that is not a model.
   */
  sha256: string | null
  /** parsed from the filename, e.g. "Q4_K_M" */
  quant: string | null
  url: string
  isMmproj: boolean
  /** part 1 of N for a split model, when applicable */
  shard: { index: number; total: number } | null
}

/**
 * One thing a person can actually choose to download: a whole model, however many files it
 * spans, or a companion to one.
 *
 * A model saved through llama.cpp's `gguf-split` is several files named
 * `<stem>-00001-of-0000N.gguf`, usually sitting together in a quant-named folder — the first of
 * them holding no tensors at all, a few megabytes of metadata. Treated as individual entries,
 * that produced one button per *file*: "part 1 of 8" at ten megabytes, with seven more clicks
 * needed before the model actually loaded, and nothing on screen said so.
 */
export interface HfVariant {
  /** The first part's path — stable, and what `hf:download` is given to identify the variant. */
  id: string
  /** What to show: the quant folder's name when there is one, else the parsed quant or filename. */
  label: string
  quant: string | null
  /** Every part's bytes, summed — the whole download, not just the first file's. */
  bytes: number
  /** Every part, in order. */
  parts: HfFile[]
  /** False when some numbered part is missing from the listing (an upload still in progress). */
  complete: boolean
  missing: number[]
  /** A model to run, its vision projector, or a separate speculative-decoding module. */
  kind: 'model' | 'mmproj' | 'mtp'
}

export interface QuantRecommendation {
  /** The `HfVariant.id` to hand back to `hf:download`. */
  variantId: string
  label: string
  reason: string
  /** predicted context if loaded now */
  predictedContext: number
  fitsFullyOnGpu: boolean
}

function authHeaders(token: string | null): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}` } : {}
}

/**
 * How long any single metadata request may take before it is abandoned.
 *
 * None of these calls had a timeout. They are awaited by bridge handlers, which are awaited by
 * the renderer, so a connection that hung — a captive portal, a stalled proxy, HuggingFace
 * having a bad day — left the Search button spinning with no error, no result and no way to
 * cancel short of restarting the app. These are small JSON reads; twenty seconds is generous.
 */
const HF_TIMEOUT_MS = 20_000

/**
 * Encode a `owner/name` repository id for use in a URL path.
 *
 * `encodeURIComponent` on the whole thing would escape the separating slash, so each segment is
 * encoded on its own. Interpolating it raw let a stray `?` or `#` in a repo id change which
 * endpoint was being addressed.
 */
function encodeRepo(repo: string): string {
  return repo.split('/').map(encodeURIComponent).join('/')
}

function baseName(p: string): string {
  const i = p.lastIndexOf('/')
  return i === -1 ? p : p.slice(i + 1)
}

function dirName(p: string): string {
  const i = p.lastIndexOf('/')
  return i === -1 ? '' : p.slice(0, i)
}

export async function searchModels(
  query: string,
  token: string | null,
  limit = 25
): Promise<HfModelSummary[]> {
  // `gguf` filter keeps results to repos that actually contain something we can run.
  const url = `${HF_API}/models?search=${encodeURIComponent(query)}&filter=gguf&sort=downloads&direction=-1&limit=${limit}&full=false`
  const res = await fetch(url, { headers: authHeaders(token), signal: AbortSignal.timeout(HF_TIMEOUT_MS) })
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
  const res = await fetch(`${HF_API}/models/${encodeRepo(repo)}/tree/main?recursive=true`, {
    headers: authHeaders(token),
    signal: AbortSignal.timeout(HF_TIMEOUT_MS)
  })
  if (res.status === 403) {
    throw new Error(`${repo} is gated. Accept its licence on huggingface.co and add a token in Settings.`)
  }
  if (!res.ok) throw new Error(`Could not list ${repo}: HTTP ${res.status}`)

  const tree = (await res.json()) as {
    path: string
    size?: number
    type: string
    lfs?: { size: number; oid?: string }
  }[]

  return tree
    .filter((f) => f.type === 'file' && f.path.toLowerCase().endsWith('.gguf'))
    .map((f) => {
      const filename = f.path
      const base = filename.split('/').pop() ?? filename
      const shardMatch = base.match(/-(\d{5})-of-(\d{5})\.gguf$/i)
      // Checked for shape rather than trusted: a checksum that is not one would fail every
      // download with a mismatch, which is a worse outcome than not verifying at all.
      const oid = f.lfs?.oid
      const sha256 = typeof oid === 'string' && /^[0-9a-f]{64}$/i.test(oid) ? oid.toLowerCase() : null

      return {
        filename,
        bytes: f.lfs?.size ?? f.size ?? 0,
        sha256,
        quant: base.match(QUANT_RE)?.[0]?.toUpperCase() ?? null,
        url: `${HF_BASE}/${repo}/resolve/main/${encodeURI(filename)}`,
        isMmproj: /mmproj/i.test(base),
        shard: shardMatch ? { index: Number(shardMatch[1]), total: Number(shardMatch[2]) } : null
      }
    })
    .sort((a, b) => a.bytes - b.bytes)
}

/**
 * Path fragments that mark a file as Unsloth's separate MTP module — the extra prediction head
 * some repos ship beside a model, in an `MTP/` folder or under an `mtp-` prefix.
 *
 * It is a companion, not a model of its own: on its own it has nowhere to attach and nothing to
 * predict from. Where a model carries its own multi-token-prediction head instead (`nextn_predict_
 * layers` in its own metadata, alongside the matching tensors — see `extractArchInfo`), that is a
 * fact about the *model* variant and is surfaced as a badge on it, not as a file to fetch.
 */
const MTP_PATH_RE = /(^|\/)mtp[-_]/i

function kindOf(f: HfFile): HfVariant['kind'] {
  if (f.isMmproj) return 'mmproj'
  if (MTP_PATH_RE.test(f.filename)) return 'mtp'
  return 'model'
}

/**
 * Group a repo's files into the variants a person actually chooses between.
 *
 * Grouping key is kind plus directory plus filename-with-the-shard-suffix-stripped: files that
 * are gguf-split's own output share a directory and a stem, so this reproduces exactly the parts
 * llama.cpp expects to find beside each other. A file with no shard suffix is a variant on its
 * own, same as it always was.
 */
export function groupVariants(files: HfFile[]): HfVariant[] {
  interface Group {
    total: number
    byIndex: Map<number, HfFile>
    kind: HfVariant['kind']
    dir: string
  }
  const groups = new Map<string, Group>()
  const out: HfVariant[] = []

  for (const f of files) {
    const kind = kindOf(f)
    if (!f.shard) {
      const dir = dirName(f.filename)
      out.push({
        id: f.filename,
        label: dir ? baseName(dir) : (f.quant ?? baseName(f.filename).replace(/\.gguf$/i, '')),
        quant: f.quant,
        bytes: f.bytes,
        parts: [f],
        complete: true,
        missing: [],
        kind
      })
      continue
    }
    const dir = dirName(f.filename)
    const shardTag = `-${String(f.shard.index).padStart(5, '0')}-of-${String(f.shard.total).padStart(5, '0')}.gguf`
    const base = baseName(f.filename)
    const stem = base.toLowerCase().endsWith(shardTag.toLowerCase()) ? base.slice(0, base.length - shardTag.length) : base
    const key = `${kind} ${dir}/${stem}`.toLowerCase()
    const group = groups.get(key) ?? { total: f.shard.total, byIndex: new Map<number, HfFile>(), kind, dir }
    group.byIndex.set(f.shard.index, f)
    groups.set(key, group)
  }

  for (const group of groups.values()) {
    const parts: HfFile[] = []
    const missing: number[] = []
    for (let i = 1; i <= group.total; i++) {
      const part = group.byIndex.get(i)
      if (part) parts.push(part)
      else missing.push(i)
    }
    const head = parts[0] ?? [...group.byIndex.values()][0]
    if (!head) continue
    out.push({
      id: head.filename,
      label: group.dir ? baseName(group.dir) : (head.quant ?? baseName(head.filename)),
      quant: head.quant,
      bytes: parts.reduce((a, p) => a + p.bytes, 0),
      parts,
      complete: missing.length === 0,
      missing,
      kind: group.kind
    })
  }

  return out.sort((a, b) => a.bytes - b.bytes)
}

/**
 * Recommend a quant for this machine.
 *
 * We do not have the GGUF header before downloading, so weights are taken from the variant's
 * real total size — every part it has, not the first shard alone — and the KV cache is estimated
 * from a typical modern GQA geometry. It is approximate by necessity; the exact fit is recomputed
 * from real metadata once the file is on disk.
 *
 * `archHints` carries whatever `peekVariantArch` has learned about a variant's header so far
 * (keyed by `HfVariant.id`). Its one effect here: a mixture-of-experts model can spread itself
 * across VRAM and system RAM at a usable speed (`--n-cpu-moe`), where a dense model spilling the
 * same way pays full price on every layer. Without a hint a variant is judged as dense, which is
 * the safe default — recommending a model as "fits" on the strength of RAM it turns out not to be
 * able to use would be the worse mistake.
 */
export function recommendQuant(
  variants: HfVariant[],
  hardware: HardwareSnapshot,
  targetContext: number,
  archHints?: Record<string, Partial<ModelArchInfo>>
): QuantRecommendation | null {
  const candidates = variants.filter((v) => v.kind === 'model' && v.complete)
  if (!candidates.length) return null

  const freeVram = hardware.gpus.reduce(
    (sum, g) => sum + (g.freeIsMeasured && g.freeVram >= 0 ? g.freeVram : g.totalVram * 0.85),
    0
  )
  // Headroom for the compute buffer, CUDA context and desktop use.
  const usableVram = Math.max(0, freeVram - 1.5 * 1024 ** 3)
  // Headroom for the OS and everything else already running, before a model's spillover eats it.
  const usableRam = Math.max(0, (hardware.freeRam ?? 0) - 4 * 1024 ** 3)

  // KV estimate: 8 KV heads x 128 dim x 2 x ~48 layers at q8_0 ≈ 105 KB per 1k tokens.
  const kvPerToken = 2 * 8 * 128 * 48 * (34 / 32)
  const kvForTarget = kvPerToken * targetContext

  const isMoe = (v: HfVariant): boolean => (archHints?.[v.id]?.expertCount ?? 0) > 0

  const ranked = candidates
    .map((v) => {
      const moe = isMoe(v)
      const ceiling = moe ? usableVram + usableRam : usableVram
      const totalNeed = v.bytes + kvForTarget
      const fits = totalNeed <= ceiling
      // Context estimate still prices the cache in VRAM: it is what the load actually uses,
      // whatever moved to system RAM is the weights, not the cache.
      const spare = usableVram - v.bytes
      const predictedContext = spare > 0 ? Math.floor(spare / kvPerToken / 1024) * 1024 : 0
      return { variant: v, moe, fits, predictedContext, quality: qualityRank(v.quant) }
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
    variantId: best.variant.id,
    label: best.variant.label,
    // "Fully on GPU" is specifically false once system RAM is part of how it fits.
    fitsFullyOnGpu: best.fits && !best.moe,
    predictedContext: Math.min(best.predictedContext, 1_048_576),
    reason: !best.fits
      ? `Nothing here fits in your ${gb(freeVram)} of free VRAM${best.moe ? ` plus ${gb(usableRam)} of usable system RAM` : ''}. ` +
        `${best.variant.label} at ${gb(best.variant.bytes)} is the closest — expect partial offload.`
      : best.moe
        ? `${best.variant.label} is ${gb(best.variant.bytes)}. Most of that is routed experts, which this app keeps ` +
          `mostly in system RAM rather than VRAM — it fits your ${gb(freeVram)} of free VRAM plus ${gb(usableRam)} ` +
          'of usable system RAM, and only a fraction of it is read for any one token.'
        : `${best.variant.label} is ${gb(best.variant.bytes)} and fits your ${gb(freeVram)} of free VRAM with room ` +
          `for roughly ${best.predictedContext.toLocaleString()} tokens of context.`
  }
}

/**
 * Higher is better quality. Ordering follows the usual llama.cpp quant hierarchy.
 *
 * The `_XL` entries are Unsloth's "Dynamic" quants — a handful of important tensors bumped to
 * higher precision within an otherwise smaller quant. Without an entry for them here they fell
 * through to the generic default (50), which ranked a Q6_K_XL file *below* a plain Q4_K_M — an
 * inversion that is easy to miss because it only bites on repos that use the naming, which by
 * volume is most of them.
 */
function qualityRank(quant: string | null): number {
  if (!quant) return 0
  const q = quant.toUpperCase()
  const table: Record<string, number> = {
    F32: 100, BF16: 95, F16: 95,
    Q8_0: 90,
    Q6_K_XL: 81, Q6_K: 80,
    Q5_K_XL: 76, Q5_K_M: 74, Q5_K_S: 72, Q5_1: 70, Q5_0: 69,
    Q4_K_XL: 65, Q4_K_M: 64, Q4_K_S: 62, Q4_1: 60, Q4_0: 58,
    IQ4_XS: 56, IQ4_NL: 55,
    Q3_K_XL: 49, Q3_K_L: 48, Q3_K_M: 46, Q3_K_S: 44,
    IQ3_M: 42, IQ3_S: 40, IQ3_XXS: 38,
    Q2_K_XL: 31, Q2_K: 30, IQ2_M: 26, IQ2_S: 24, IQ2_XXS: 22,
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

/**
 * Peek a variant's own architecture facts before downloading it, by range-reading its header
 * over HTTP rather than fetching the whole file.
 *
 * Only the first part is read. For a split model that is the part `gguf-split` always writes
 * the whole metadata block into — which is why Unsloth's shard 1 is a few megabytes on its own —
 * and for a single-file model it is the model. This is what tells Discover a variant is a
 * mixture-of-experts model (worth recommending even when it does not fit in VRAM alone) and
 * whether it carries its own multi-token-prediction head, before a single gigabyte of weights
 * has moved. A failure here costs nothing but a missing badge — real placement is always
 * recomputed from the real file once it is on disk.
 */
export async function peekVariantArch(
  variant: Pick<HfVariant, 'parts'>,
  token: string | null
): Promise<Partial<ModelArchInfo> | null> {
  const head = variant.parts[0]
  if (!head) return null
  // Enough for the KV block and tensor directory of almost any model: the huge arrays (the
  // vocabulary chief among them) are elided by the parser rather than read in full, and 1224
  // tensors' worth of directory entries — Qwen3.8-Flash-Next's count — is under 100 KB.
  const peekBytes = Math.min(Math.max(head.bytes || 4 * 1024 * 1024, 1), 4 * 1024 * 1024)
  let tmp: string | null = null
  try {
    const res = await fetch(head.url, {
      headers: { ...authHeaders(token), Range: `bytes=0-${peekBytes - 1}` },
      signal: AbortSignal.timeout(HF_TIMEOUT_MS)
    })
    if (!res.ok && res.status !== 206) return null
    const buf = Buffer.from(await res.arrayBuffer())
    if (buf.length < 8 || buf.readUInt32LE(0) !== 0x46554747) return null

    // Parsed via the real reader rather than duplicated here: writing the range to a temp file
    // and handing it to `readGguf` means this can never disagree with what the library scan sees
    // once the file actually lands.
    tmp = path.join(os.tmpdir(), `llmm-peek-${crypto.randomBytes(6).toString('hex')}.gguf`)
    await fsp.writeFile(tmp, buf)
    const meta = await readGguf(tmp)
    const arch = extractArchInfo(meta)
    return { mtpLayers: arch.mtpLayers, expertCount: arch.expertCount, expertUsedCount: arch.expertUsedCount }
  } catch {
    return null
  } finally {
    if (tmp) await fsp.rm(tmp, { force: true }).catch(() => undefined)
  }
}

export function estimateArchFromSize(bytes: number): Partial<ModelArchInfo> {
  // Only used for display before download; real values come from the header afterwards.
  return { weightBytes: bytes }
}
