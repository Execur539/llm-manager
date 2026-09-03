/**
 * Multimodal message construction.
 *
 * Images and audio go through llama.cpp's OpenAI-compatible content-part format. Video is the
 * interesting case: llama.cpp expands a video into frames internally via an FFmpeg subprocess,
 * and that support is model-agnostic — any vision model can consume it.
 *
 * How many frames a video gets is derived from the model's own context window rather than fixed,
 * and where they are taken from prefers scene changes when the sample is sparse enough for that
 * to matter. Where the server can decode video, the chosen frames are re-encoded as a short clip
 * and sent as one — half the tokens of loose images, and the only way the model sees the order.
 * See video-plan.ts for the arithmetic.
 */

import path from 'node:path'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import crypto from 'node:crypto'
import { spawn } from 'node:child_process'
import type { ContentPart } from '../runtime/llama'
import type { ModelCapabilities } from '@shared/types'
import { planVideo, chooseTimestamps, type VideoDetail } from './video-plan'
import { runtimeBinary } from '../runtime/binaries'
import { TOOL_OUTPUT_DIR } from '../storage/paths'
import { classifyAttachment, toDataUrl } from './repo'
import { extractText } from '../rag/index'

/**
 * How much of a document to inline.
 *
 * Generous enough for a source file or a README, small enough that dropping a 5 MB log does not
 * silently eat the whole context window and push the conversation out of it. Anything larger is
 * truncated with a note saying so, because a quietly shortened file is worse than a refusal.
 */
const MAX_DOC_CHARS = 60_000

/** Fence language, so inlined code is highlighted rather than rendered as prose. */
const LANGUAGE_BY_EXT: Record<string, string> = {
  '.ts': 'typescript',
  '.tsx': 'tsx',
  '.js': 'javascript',
  '.jsx': 'jsx',
  '.py': 'python',
  '.rs': 'rust',
  '.go': 'go',
  '.java': 'java',
  '.c': 'c',
  '.h': 'c',
  '.cpp': 'cpp',
  '.cs': 'csharp',
  '.rb': 'ruby',
  '.php': 'php',
  '.swift': 'swift',
  '.kt': 'kotlin',
  '.sh': 'bash',
  '.ps1': 'powershell',
  '.sql': 'sql',
  '.json': 'json',
  '.yaml': 'yaml',
  '.yml': 'yaml',
  '.toml': 'toml',
  '.xml': 'xml',
  '.html': 'html',
  '.css': 'css',
  '.md': 'markdown'
}

export interface AttachmentPlan {
  parts: ContentPart[]
  notes: string[]
  /** frames we extracted ourselves, when we did the sampling rather than llama.cpp */
  extractedFrames: string[]
}

interface VideoProbe {
  duration: number
  /** Source frame rate, needed to pick a tolerance when selecting exact moments. */
  fps: number
  /** width / height */
  aspect: number
}

function ffprobePath(): string | null {
  const p = runtimeBinary('ffmpeg').replace(/ffmpeg(\.exe)?$/i, 'ffprobe$1')
  return fs.existsSync(p) ? p : null
}

/** Run a bundled tool, returning its output, with a hard ceiling on how long it may take. */
function runTool(exe: string, args: string[], timeoutMs: number): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const child = spawn(exe, args, { windowsHide: true })
    const timer = setTimeout(() => child.kill('SIGKILL'), timeoutMs)
    let stdout = ''
    let stderr = ''
    child.stdout?.on('data', (d: Buffer) => (stdout += d.toString()))
    // Read stderr rather than letting it fill: ffmpeg reports frame metadata there, and an
    // unread pipe that fills stops the child dead.
    child.stderr?.on('data', (d: Buffer) => (stderr += d.toString()))
    child.on('close', () => {
      clearTimeout(timer)
      resolve({ stdout, stderr })
    })
    child.on('error', () => {
      clearTimeout(timer)
      resolve({ stdout: '', stderr: '' })
    })
  })
}

/** Duration, frame rate and shape, in one metadata read. */
async function probeVideo(file: string): Promise<VideoProbe | null> {
  const exe = ffprobePath()
  if (!exe) return null
  const { stdout } = await runTool(
    exe,
    [
      '-v', 'error',
      '-select_streams', 'v:0',
      '-show_entries', 'format=duration:stream=width,height,avg_frame_rate',
      '-of', 'json',
      file
    ],
    30_000
  )
  try {
    const j = JSON.parse(stdout) as {
      format?: { duration?: string }
      streams?: { width?: number; height?: number; avg_frame_rate?: string }[]
    }
    const duration = Number(j.format?.duration ?? 0)
    const s = j.streams?.[0]
    const [num, den] = (s?.avg_frame_rate ?? '0/1').split('/').map(Number)
    const fps = den > 0 && num > 0 ? num / den : 30
    const aspect = s?.width && s?.height ? s.width / s.height : 16 / 9
    if (!Number.isFinite(duration) || duration <= 0) return null
    return { duration, fps, aspect }
  } catch {
    return null
  }
}

/**
 * Timestamps where the picture changes materially.
 *
 * ffmpeg's scene score compares each frame with the one before it, so the frames it selects are
 * the cuts. This matters most where the budget is thin: sampling a lecture uniformly spends most
 * of the frames on the same slide, while the moments that carry information are exactly the ones
 * that differ from their predecessor.
 *
 * Failure is not an error. Scene detection is an optimisation on top of uniform coverage, so a
 * codec it cannot read, or a timeout, falls back to spacing frames evenly.
 */
async function detectScenes(file: string, threshold = 0.3): Promise<number[]> {
  const exe = runtimeBinary('ffmpeg')
  if (!fs.existsSync(exe)) return []
  const { stderr } = await runTool(
    exe,
    [
      '-hide_banner',
      '-i', file,
      // Downscaled first: the score only needs the gist, and comparing full-resolution frames
      // over a ten-minute video is far slower than the sampling it is meant to inform.
      '-vf', `scale=192:-2,select='gt(scene,${threshold})',showinfo`,
      '-f', 'null',
      '-'
    ],
    120_000
  )
  const times: number[] = []
  for (const m of stderr.matchAll(/pts_time:([0-9.]+)/g)) {
    const t = Number(m[1])
    if (Number.isFinite(t)) times.push(t)
  }
  return times
}

/**
 * Pull specific moments out of a video in a single pass.
 *
 * One invocation with a `select` expression rather than one seek per frame: at a few hundred
 * frames the per-process overhead of seeking individually dominates everything else. The
 * tolerance is derived from the source frame rate so that each requested moment matches exactly
 * one frame — too tight and moments are missed, too loose and neighbours are duplicated.
 */
async function extractAt(file: string, times: number[], width: number): Promise<string[]> {
  const exe = runtimeBinary('ffmpeg')
  const dir = path.join(TOOL_OUTPUT_DIR, `frames-${crypto.randomBytes(4).toString('hex')}`)
  await fsp.mkdir(dir, { recursive: true })

  const probe = await probeVideo(file)
  const tolerance = 0.5 / Math.max(1, probe?.fps ?? 30)
  const expr = times.map((t) => `lt(abs(t-${t.toFixed(3)})\\,${tolerance.toFixed(4)})`).join('+')

  await runTool(
    exe,
    [
      '-hide_banner',
      '-i', file,
      '-vf', `select='${expr}',scale='min(${width},iw)':-2`,
      // Keep every selected frame; the default would re-time them to a constant rate and drop some.
      '-vsync', '0',
      '-q:v', '3',
      path.join(dir, 'frame-%04d.jpg')
    ],
    300_000
  )
  const files = (await fsp.readdir(dir)).filter((f) => f.endsWith('.jpg')).sort()
  return files.map((f) => path.join(dir, f))
}

/** Evenly spaced frames, for when the sample is dense enough that scene detection adds nothing. */
async function extractUniform(file: string, count: number, duration: number, width: number): Promise<string[]> {
  const exe = runtimeBinary('ffmpeg')
  const dir = path.join(TOOL_OUTPUT_DIR, `frames-${crypto.randomBytes(4).toString('hex')}`)
  await fsp.mkdir(dir, { recursive: true })
  const fps = duration > 0 ? count / duration : 1
  await runTool(
    exe,
    [
      '-hide_banner',
      '-i', file,
      '-vf', `fps=${fps.toFixed(5)},scale='min(${width},iw)':-2`,
      '-frames:v', String(count),
      '-q:v', '3',
      path.join(dir, 'frame-%04d.jpg')
    ],
    300_000
  )
  const files = (await fsp.readdir(dir)).filter((f) => f.endsWith('.jpg')).sort()
  return files.map((f) => path.join(dir, f))
}

/**
 * Re-encode chosen frames as a short video, so the server can treat them as one.
 *
 * This is what lets us keep control of sampling while still using llama.cpp's video path. Sent as
 * loose images the frames cost full price and carry no order; sent as a video they are paired by
 * a 3D convolution, positioned by temporal M-RoPE and given timestamps — half the tokens and an
 * actual sense of sequence.
 *
 * Encoded at 2 fps deliberately. The server re-samples video at its own fixed rate (4 fps in this
 * build, with no flag to change it), and it cannot sample faster than the source — so a source
 * below that rate is taken whole. Encoding at the original rate would have the server throw most
 * of these frames away again.
 */
async function encodeCondensed(frames: string[], dir: string): Promise<string | null> {
  if (!frames.length) return null
  const exe = runtimeBinary('ffmpeg')
  const listFile = path.join(dir, 'frames.txt')
  // Concat demuxer rather than a numbered pattern: the frames are already named in order and
  // this avoids re-deriving a glob that has to match exactly.
  await fsp.writeFile(
    listFile,
    frames.map((f) => `file '${f.replace(/\\/g, '/')}'\nduration 0.5`).join('\n') +
      `\nfile '${frames[frames.length - 1].replace(/\\/g, '/')}'\n`,
    'utf8'
  )
  const out = path.join(dir, 'condensed.mp4')
  await runTool(
    exe,
    ['-hide_banner', '-y', '-f', 'concat', '-safe', '0', '-i', listFile, '-r', '2', '-pix_fmt', 'yuv420p', out],
    300_000
  )
  return fs.existsSync(out) ? out : null
}

/**
 * Sample a video according to the model's actual context budget.
 *
 * Returns the frames and, when the server can take video, a condensed clip built from them.
 */
export async function sampleVideo(
  file: string,
  opts: { contextLength: number; share: number; detail: VideoDetail; temporalPairing: boolean }
): Promise<{ frames: string[]; condensed: string | null; note: string }> {
  const probe = await probeVideo(file)
  const duration = probe?.duration ?? 0
  const plan = planVideo({
    contextLength: opts.contextLength,
    share: opts.share,
    detail: opts.detail,
    durationSeconds: duration,
    aspect: probe?.aspect ?? 16 / 9,
    temporalPairing: opts.temporalPairing
  })

  /*
   * Scene detection earns its cost only when the sample is sparse.
   *
   * Above roughly one frame a second the uniform sample already lands on every cut, and a second
   * full decode of the video to prove it is pure overhead. Below that, the cuts are exactly what
   * a uniform sample misses.
   */
  const dense = duration > 0 && plan.count / duration >= 1
  let frames: string[]
  let how: string

  if (dense || duration <= 0) {
    frames = await extractUniform(file, plan.count, duration, plan.width)
    how = 'evenly spaced'
  } else {
    const scenes = await detectScenes(file)
    const times = chooseTimestamps(scenes, duration, plan.count)
    frames = times.length ? await extractAt(file, times, plan.width) : await extractUniform(file, plan.count, duration, plan.width)
    how = scenes.length ? `${scenes.length} scene changes, then the widest gaps` : 'evenly spaced'
  }

  let condensed: string | null = null
  if (opts.temporalPairing && frames.length) {
    condensed = await encodeCondensed(frames, path.dirname(frames[0]))
  }

  const mins = Math.floor(duration / 60)
  const secs = Math.round(duration % 60)
  const note =
    `sampled ${frames.length} frames from ${mins}m ${secs}s ` +
    `(${plan.effectiveFps.toFixed(2)} fps at ${plan.width}px, ${how}; ` +
    `about ${plan.estimatedTokens.toLocaleString()} tokens)`

  return { frames, condensed, note }
}

/**
 * Turn a user message plus attachments into multimodal content parts, refusing clearly
 * (rather than silently dropping) anything the loaded model cannot accept.
 */
/**
 * What a video may spend, and whether the server can take one.
 *
 * Passed in rather than read here: the context length belongs to the loaded plan and the server's
 * modalities are a property of the running process, neither of which this module should be
 * reaching for. Defaults keep the older two-argument callers working.
 */
export interface VideoContext {
  contextLength: number
  share: number
  detail: VideoDetail
  /** True when the running server reports it can decode video, not merely that the model was trained on it. */
  serverTakesVideo: boolean
}

const DEFAULT_VIDEO: VideoContext = {
  contextLength: 8192,
  share: 0.45,
  detail: 'balanced',
  serverTakesVideo: false
}

export async function buildContent(
  text: string,
  attachments: string[],
  caps: ModelCapabilities,
  video: VideoContext = DEFAULT_VIDEO
): Promise<AttachmentPlan> {
  const parts: ContentPart[] = []
  const notes: string[] = []
  const extractedFrames: string[] = []

  if (text.trim()) parts.push({ type: 'text', text })

  for (const file of attachments) {
    const kind = classifyAttachment(file)

    if (kind === 'image') {
      if (!caps.vision) {
        notes.push(`${path.basename(file)} skipped: the loaded model has no vision projector (mmproj).`)
        continue
      }
      parts.push({ type: 'image_url', image_url: { url: await toDataUrl(file) } })
      continue
    }

    if (kind === 'audio') {
      if (!caps.audio) {
        notes.push(`${path.basename(file)} skipped: the loaded model does not accept audio input.`)
        continue
      }
      const buf = await fsp.readFile(file)
      parts.push({
        type: 'input_audio',
        input_audio: { data: buf.toString('base64'), format: path.extname(file).slice(1) || 'wav' }
      })
      continue
    }

    if (kind === 'video') {
      if (!caps.videoPossible) {
        notes.push(`${path.basename(file)} skipped: the loaded model has no vision projector, so video cannot be read.`)
        continue
      }
      try {
        const sampled = await sampleVideo(file, {
          contextLength: video.contextLength,
          share: video.share,
          detail: video.detail,
          temporalPairing: video.serverTakesVideo
        })
        extractedFrames.push(...sampled.frames)

        if (video.serverTakesVideo && sampled.condensed) {
          /*
           * One video part, not N image parts.
           *
           * The server pairs consecutive frames with a 3D convolution and positions them with
           * temporal M-RoPE, which halves the token cost and is the only way the model learns
           * the order things happened in. The frames were still chosen here, because the server
           * samples at a fixed rate with no flag to change it — at 4 fps a ten-minute video is
           * 2,400 frames, several times the whole context window.
           */
          const data = await fsp.readFile(sampled.condensed)
          parts.push({ type: 'input_video', input_video: { data: data.toString('base64') } })
          extractedFrames.push(sampled.condensed)
          notes.push(`${path.basename(file)}: ${sampled.note}, sent as video so the model can follow the order.`)
        } else {
          for (const frame of sampled.frames) {
            parts.push({ type: 'image_url', image_url: { url: await toDataUrl(frame) } })
          }
          notes.push(
            `${path.basename(file)}: ${sampled.note}, sent as stills.` +
              (caps.nativeVideo
                ? ' This build cannot take video directly, so the frames carry no timing.'
                : ' This model was not trained on video, so treat the result as a description of stills.')
          )
        }
      } catch (err) {
        notes.push(`${path.basename(file)} could not be processed: ${err instanceof Error ? err.message : String(err)}`)
      }
      continue
    }

    /*
     * Documents are inlined as text.
     *
     * This used to refuse them and point at the document collections instead, which is the right
     * answer for a corpus you will ask about repeatedly and the wrong one for "look at this file"
     * — the common case, and the one that made attaching a source file feel broken.
     */
    const name = path.basename(file)
    try {
      const raw = await extractText(file)
      const trimmed = raw.trim()

      if (!trimmed) {
        notes.push(`${name} contained no readable text.`)
        continue
      }

      const truncated = trimmed.length > MAX_DOC_CHARS
      const body = truncated ? trimmed.slice(0, MAX_DOC_CHARS) : trimmed
      const lang = LANGUAGE_BY_EXT[path.extname(file).toLowerCase()] ?? ''

      parts.push({
        type: 'text',
        text: `Attached file \`${name}\`:\n\n\`\`\`${lang}\n${body}\n\`\`\``
      })

      if (truncated) {
        notes.push(
          `${name} is ${Math.round(trimmed.length / 1000)}k characters; the first ` +
            `${Math.round(MAX_DOC_CHARS / 1000)}k were included. Add it to a document collection to ` +
            `search the whole thing instead.`
        )
      }
    } catch (err) {
      notes.push(`${name} could not be read: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  return { parts, notes, extractedFrames }
}
