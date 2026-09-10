/**
 * Model library: scans the models directory, parses GGUF metadata, detects capabilities,
 * and caches the result so a library of 40 GB files doesn't get re-parsed on every launch.
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import crypto from 'node:crypto'
import type { ModelCapabilities, ModelRecord, ReasoningSupport } from '@shared/types'
import { extractArchInfo, readGguf, readGgufParts, templateSupportsTools } from './gguf'
import { detectReasoning, NO_REASONING } from './reasoning'
import { APPDATA_DIR } from '../storage/paths'

const INDEX_FILE = path.join(APPDATA_DIR, 'model-index.json')

/**
 * Bump when the shape of a cached record changes.
 *
 * The index exists so a library of 40 GB files is not re-parsed on every launch, and it keys on
 * path/size/mtime — none of which change when *this code* starts recording something new. Without
 * a schema version, adding a capability leaves every already-scanned model with that field
 * missing, and the gap only shows up wherever the UI happens to read it.
 */
const INDEX_SCHEMA = 3

interface IndexEntry {
  path: string
  size: number
  mtimeMs: number
  record: ModelRecord
}

interface IndexFile {
  schema: number
  entries: IndexEntry[]
}

/**
 * Models known to be trained on video, as opposed to merely accepting frames.
 * llama.cpp expands video into frames for *any* vision model, so this list only affects
 * how we describe the model and what frame-sampling defaults we choose — never whether
 * video is offered at all. Matched loosely against architecture and model name.
 */
const NATIVE_VIDEO_HINTS = [/qwen3\.?8/i, /qwen3-?vl/i, /qwen2\.5-?vl/i, /omni/i, /video/i]

/**
 * Quant advertised by the filename. Handles plain quants (Q4_K_M) and vendor-prefixed
 * mixed-precision variants (UD-Q4_K_XL, i1-Q4_K_S).
 */
const FILENAME_QUANT_RE = /(?:^|[-_.])((?:UD|i1|IQ)?[-_]?(?:IQ\d[A-Z_]*|Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32))(?=[-_.]|$)/i

export function quantFromFilename(filename: string): string | null {
  const base = filename.replace(/\.gguf$/i, '')
  const matches = [...base.matchAll(new RegExp(FILENAME_QUANT_RE, 'gi'))]
  if (!matches.length) return null
  // The quant is conventionally the last such token in the name.
  return matches[matches.length - 1][1].replace(/^[-_]/, '').toUpperCase()
}

function idFor(filePath: string): string {
  return crypto.createHash('sha1').update(filePath.toLowerCase()).digest('hex').slice(0, 16)
}

async function loadIndex(): Promise<Map<string, IndexEntry>> {
  try {
    const raw = JSON.parse(await fsp.readFile(INDEX_FILE, 'utf8')) as IndexFile | IndexEntry[]

    // A bare array is the pre-versioned format: discard it and re-parse rather than serve
    // records that predate whatever the current code expects to find on them.
    if (Array.isArray(raw) || raw.schema !== INDEX_SCHEMA) return new Map()

    return new Map(raw.entries.map((e) => [e.path, e]))
  } catch {
    return new Map()
  }
}

async function saveIndex(entries: IndexEntry[]): Promise<void> {
  await fsp.mkdir(APPDATA_DIR, { recursive: true })
  await fsp.writeFile(INDEX_FILE, JSON.stringify({ schema: INDEX_SCHEMA, entries }, null, 2))
}

/** Walk the models dir for .gguf files, skipping the partial-download staging area. */
async function findGgufFiles(root: string): Promise<string[]> {
  const out: string[] = []
  async function walk(dir: string): Promise<void> {
    let entries: fs.Dirent[]
    try {
      entries = await fsp.readdir(dir, { withFileTypes: true })
    } catch {
      return
    }
    for (const e of entries) {
      if (e.name.startsWith('.')) continue // .partial and friends
      const full = path.join(dir, e.name)
      if (e.isDirectory()) await walk(full)
      else if (e.isFile() && e.name.toLowerCase().endsWith('.gguf')) out.push(full)
    }
  }
  await walk(root)
  return out
}

/** `<name>-00001-of-00004.gguf`: the naming llama.cpp's gguf-split writes, and finds the other parts by. */
const SHARD_RE = /-(\d{5})-of-(\d{5})\.gguf$/i

/*
 * A speculative-decoding module shipped beside a model — Unsloth's `mtp-*.gguf` for
 * Qwen3.8-Flash-Next, for one. It holds only the prediction head, so it is part of a model rather
 * than one, and listed as a model it would only fail to load.
 */
const DRAFT_MODULE_RE = /^mtp[-_]/i

interface ModelEntry {
  /** What llama.cpp is given: the file itself, or the first part of a split model. */
  head: string
  /** Every part present, in order. */
  parts: string[]
  /** Part numbers the split names that are not on disk. */
  missing: number[]
  total: number
}

/**
 * The models in a list of GGUF files: one entry per model, however many files it spans.
 *
 * Every file used to be its own entry, so a model in eight parts appeared eight times — part one
 * holding no tensors at all and reporting its size as a few megabytes, the other seven with no
 * architecture and an error each.
 */
function modelEntries(files: string[]): ModelEntry[] {
  const out: ModelEntry[] = []
  const groups = new Map<string, { total: number; byIndex: Map<number, string> }>()
  for (const file of files) {
    const base = path.basename(file)
    // Companions, not models: projectors pair with a model, and draft modules are a piece of one.
    if (/mmproj/i.test(base) || DRAFT_MODULE_RE.test(base)) continue
    const shard = SHARD_RE.exec(base)
    if (!shard) {
      out.push({ head: file, parts: [file], missing: [], total: 1 })
      continue
    }
    const key = path.join(path.dirname(file), base.slice(0, shard.index)).toLowerCase()
    const group = groups.get(key) ?? { total: Number(shard[2]), byIndex: new Map<number, string>() }
    group.byIndex.set(Number(shard[1]), file)
    groups.set(key, group)
  }
  for (const group of groups.values()) {
    const parts: string[] = []
    const missing: number[] = []
    for (let i = 1; i <= group.total; i++) {
      const part = group.byIndex.get(i)
      if (part) parts.push(part)
      else missing.push(i)
    }
    out.push({ head: group.byIndex.get(1) ?? parts[0], parts, missing, total: group.total })
  }
  return out
}

/**
 * An mmproj file is the multimodal projector that pairs with a text model.
 * Convention is a sibling file with "mmproj" in the name; we match within the same folder
 * and fall back to any mmproj in the model's directory.
 */
function findMmproj(modelPath: string, allFiles: string[]): string | null {
  const dir = path.dirname(modelPath)
  const base = path.basename(modelPath, '.gguf').toLowerCase()

  const candidates = allFiles.filter(
    (f) => path.dirname(f) === dir && /mmproj/i.test(path.basename(f)) && f !== modelPath
  )
  if (!candidates.length) return null

  // Prefer one whose name shares the model's stem.
  const stem = base.replace(/[-_.](q\d.*|f16|f32|bf16)$/i, '')
  const exact = candidates.find((c) => path.basename(c).toLowerCase().includes(stem.slice(0, 12)))
  return exact ?? candidates[0]
}

async function detectCapabilities(
  modelPath: string,
  archName: string,
  modelName: string | null,
  mmproj: string | null,
  toolsFromTemplate: boolean,
  reasoning: ReasoningSupport = NO_REASONING
): Promise<ModelCapabilities> {
  let vision = false
  let audio = false

  if (mmproj) {
    try {
      const meta = await readGguf(mmproj)
      const kv = meta.kv
      // clip.* keys indicate a vision projector; whisper/audio keys indicate audio.
      const keys = Object.keys(kv).join(' ').toLowerCase()
      vision = keys.includes('clip.vision') || keys.includes('clip.has_vision_encoder') || keys.includes('clip.')
      audio = keys.includes('clip.has_audio_encoder') || keys.includes('audio') || keys.includes('whisper')
      /*
       * The declared value wins over the presence of the key.
       *
       * The guesses above match key *names*, so a projector carrying
       * `clip.has_audio_encoder = false` — a vision-only one being explicit about it — was read
       * as accepting audio, because the substring is there either way. Vision had this
       * correction; audio only had the half that turns it on, so the model advertised an input
       * it would reject.
       */
      if (kv['clip.has_vision_encoder'] === false) vision = false
      if (kv['clip.has_vision_encoder'] === true) vision = true
      if (kv['clip.has_audio_encoder'] === false) audio = false
      if (kv['clip.has_audio_encoder'] === true) audio = true
    } catch {
      // An unreadable mmproj still signals *some* multimodal intent; assume vision.
      vision = true
    }
  }

  const haystack = `${archName} ${modelName ?? ''} ${path.basename(modelPath)}`
  const nativeVideo = vision && NATIVE_VIDEO_HINTS.some((re) => re.test(haystack))

  return {
    vision,
    audio,
    nativeVideo,
    // llama.cpp expands video into frames for any vision model, so vision implies video is reachable.
    videoPossible: vision,
    tools: toolsFromTemplate,
    mmprojPath: mmproj,
    reasoning
  }
}

export async function scanLibrary(modelsDir: string): Promise<ModelRecord[]> {
  if (!fs.existsSync(modelsDir)) return []

  const index = await loadIndex()
  const files = await findGgufFiles(modelsDir)

  const records: ModelRecord[] = []
  const nextIndex: IndexEntry[] = []

  for (const entry of modelEntries(files)) {
    const file = entry.head
    const split = entry.total > 1
    const stats: fs.Stats[] = []
    try {
      for (const part of entry.parts) stats.push(await fsp.stat(part))
    } catch {
      continue
    }
    // One signature for the whole set, so a part landing or changing re-parses the model.
    const size = stats.reduce((a, s) => a + s.size, 0)
    const mtimeMs = Math.max(...stats.map((s) => s.mtimeMs))
    const st = stats[0]

    const cached = index.get(file)
    if (cached && cached.size === size && cached.mtimeMs === mtimeMs) {
      records.push(cached.record)
      nextIndex.push(cached)
      continue
    }

    const mmproj = findMmproj(file, files)
    let record: ModelRecord
    try {
      if (entry.missing.length) {
        throw new Error(
          `Part${entry.missing.length === 1 ? '' : 's'} ${entry.missing.join(', ')} of ${entry.total} ` +
            `${entry.missing.length === 1 ? 'is' : 'are'} missing — the download may still be in progress.`
        )
      }
      const { meta, weightBytes } = split
        ? await readGgufParts(entry.parts)
        : { meta: await readGguf(file), weightBytes: undefined }
      const arch = extractArchInfo(meta, split ? undefined : st.size, weightBytes)
      const caps = await detectCapabilities(
        file,
        arch.architecture,
        arch.name,
        mmproj,
        templateSupportsTools(meta),
        detectReasoning(meta.kv['tokenizer.chat_template'] as string | undefined)
      )
      const quantLabel = quantFromFilename(path.basename(file))
      record = {
        id: idFor(file),
        repo: null,
        filename: path.basename(file),
        path: file,
        bytes: size,
        ...(split ? { parts: entry.parts } : {}),
        arch,
        caps,
        addedAt: st.birthtimeMs || Date.now(),
        lastUsedAt: null,
        favourite: false,
        quantLabel,
        // Mixed-precision quants advertise one thing and are mostly another; flag the mismatch
        // rather than silently showing whichever we happened to compute.
        mixedQuant: !!quantLabel && !quantLabel.includes(arch.quant) && !arch.quant.includes(quantLabel),
        tags: autoTags(quantLabel ?? arch.quant, arch.contextLength, caps, size)
      }
    } catch (err) {
      record = {
        id: idFor(file),
        repo: null,
        filename: path.basename(file),
        path: file,
        bytes: size,
        ...(split ? { parts: entry.parts } : {}),
        arch: null,
        caps: {
          vision: false,
          audio: false,
          nativeVideo: false,
          videoPossible: false,
          tools: false,
          mmprojPath: mmproj,
          reasoning: NO_REASONING
        },
        addedAt: Date.now(),
        lastUsedAt: null,
        favourite: false,
        quantLabel: quantFromFilename(path.basename(file)),
        mixedQuant: false,
        tags: [],
        error: err instanceof Error ? err.message : String(err)
      }
    }

    // Preserve user-set fields across a re-parse.
    if (cached) {
      record.favourite = cached.record.favourite
      record.tags = Array.from(new Set([...record.tags, ...cached.record.tags.filter((t) => !t.startsWith('auto:'))]))
      record.lastUsedAt = cached.record.lastUsedAt
      record.repo = cached.record.repo
    }

    records.push(record)
    nextIndex.push({ path: file, size, mtimeMs, record })
  }

  await saveIndex(nextIndex)
  return records
}

function autoTags(quant: string, ctx: number, caps: ModelCapabilities, bytes: number): string[] {
  const tags = [`auto:${quant}`]
  const gb = bytes / (1024 ** 3)
  tags.push(`auto:${gb < 5 ? 'small' : gb < 20 ? 'medium' : 'large'}`)
  if (ctx >= 131072) tags.push('auto:128k+')
  else if (ctx >= 65536) tags.push('auto:64k+')
  if (caps.vision) tags.push('auto:vision')
  if (caps.audio) tags.push('auto:audio')
  if (caps.nativeVideo) tags.push('auto:video')
  if (caps.tools) tags.push('auto:tools')
  return tags
}

/** Disk usage summary for the library page. */
export async function libraryDiskUsage(modelsDir: string): Promise<{
  totalBytes: number
  fileCount: number
  partialBytes: number
  freeBytes: number
}> {
  let totalBytes = 0
  let fileCount = 0
  let partialBytes = 0

  const files = fs.existsSync(modelsDir) ? await findGgufFiles(modelsDir) : []
  for (const f of files) {
    try {
      const st = await fsp.stat(f)
      totalBytes += st.size
      fileCount++
    } catch {
      /* skip unreadable */
    }
  }

  const partialDir = path.join(modelsDir, '.partial')
  if (fs.existsSync(partialDir)) {
    for (const f of await fsp.readdir(partialDir)) {
      try {
        partialBytes += (await fsp.stat(path.join(partialDir, f))).size
      } catch {
        /* skip */
      }
    }
  }

  let freeBytes = 0
  try {
    const st = await fsp.statfs(modelsDir)
    freeBytes = st.bavail * st.bsize
  } catch {
    freeBytes = -1
  }

  return { totalBytes, fileCount, partialBytes, freeBytes }
}
