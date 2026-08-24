/**
 * Model library: scans the models directory, parses GGUF metadata, detects capabilities,
 * and caches the result so a library of 40 GB files doesn't get re-parsed on every launch.
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import crypto from 'node:crypto'
import type { ModelCapabilities, ModelRecord } from '@shared/types'
import { extractArchInfo, readGguf, templateSupportsTools } from './gguf'
import { APPDATA_DIR } from '../storage/paths'

const INDEX_FILE = path.join(APPDATA_DIR, 'model-index.json')

interface IndexEntry {
  path: string
  size: number
  mtimeMs: number
  record: ModelRecord
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
    const raw = JSON.parse(await fsp.readFile(INDEX_FILE, 'utf8')) as IndexEntry[]
    return new Map(raw.map((e) => [e.path, e]))
  } catch {
    return new Map()
  }
}

async function saveIndex(entries: IndexEntry[]): Promise<void> {
  await fsp.mkdir(APPDATA_DIR, { recursive: true })
  await fsp.writeFile(INDEX_FILE, JSON.stringify(entries, null, 2))
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
  toolsFromTemplate: boolean
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
      if (kv['clip.has_vision_encoder'] === false) vision = false
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
    mmprojPath: mmproj
  }
}

export async function scanLibrary(modelsDir: string): Promise<ModelRecord[]> {
  if (!fs.existsSync(modelsDir)) return []

  const index = await loadIndex()
  const files = await findGgufFiles(modelsDir)
  // mmproj files are companions, not standalone entries.
  const modelFiles = files.filter((f) => !/mmproj/i.test(path.basename(f)))

  const records: ModelRecord[] = []
  const nextIndex: IndexEntry[] = []

  for (const file of modelFiles) {
    let st: fs.Stats
    try {
      st = await fsp.stat(file)
    } catch {
      continue
    }

    const cached = index.get(file)
    if (cached && cached.size === st.size && cached.mtimeMs === st.mtimeMs) {
      records.push(cached.record)
      nextIndex.push(cached)
      continue
    }

    const mmproj = findMmproj(file, files)
    let record: ModelRecord
    try {
      const meta = await readGguf(file)
      const arch = extractArchInfo(meta, st.size)
      const caps = await detectCapabilities(file, arch.architecture, arch.name, mmproj, templateSupportsTools(meta))
      const quantLabel = quantFromFilename(path.basename(file))
      record = {
        id: idFor(file),
        repo: null,
        filename: path.basename(file),
        path: file,
        bytes: st.size,
        arch,
        caps,
        addedAt: st.birthtimeMs || Date.now(),
        lastUsedAt: null,
        favourite: false,
        quantLabel,
        // Mixed-precision quants advertise one thing and are mostly another; flag the mismatch
        // rather than silently showing whichever we happened to compute.
        mixedQuant: !!quantLabel && !quantLabel.includes(arch.quant) && !arch.quant.includes(quantLabel),
        tags: autoTags(quantLabel ?? arch.quant, arch.contextLength, caps, st.size)
      }
    } catch (err) {
      record = {
        id: idFor(file),
        repo: null,
        filename: path.basename(file),
        path: file,
        bytes: st.size,
        arch: null,
        caps: { vision: false, audio: false, nativeVideo: false, videoPossible: false, tools: false, mmprojPath: mmproj },
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
    nextIndex.push({ path: file, size: st.size, mtimeMs: st.mtimeMs, record })
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
