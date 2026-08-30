/**
 * Multimodal message construction.
 *
 * Images and audio go through llama.cpp's OpenAI-compatible content-part format. Video is the
 * interesting case: llama.cpp expands a video into frames internally via an FFmpeg subprocess,
 * and that support is model-agnostic — any vision model can consume it.
 *
 * We still distinguish natively-video-trained models (Qwen3.8, Qwen3-VL, Omni) from ones that
 * merely accept frames, because the right frame count differs a lot between the two. A model
 * trained on video handles a dense sample; a still-image model does better with a handful of
 * well-spaced keyframes.
 */

import path from 'node:path'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import crypto from 'node:crypto'
import { spawn } from 'node:child_process'
import type { ContentPart } from '../runtime/llama'
import type { ModelCapabilities } from '@shared/types'
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

/** Frame budget. Dense for models trained on video; sparse for still-image models. */
function frameBudget(caps: ModelCapabilities): { count: number; reason: string } {
  return caps.nativeVideo
    ? { count: 16, reason: 'model is trained on video, so a dense sample is useful' }
    : { count: 6, reason: 'model handles still images, so a sparse keyframe sample reads better' }
}

async function videoDurationSeconds(file: string): Promise<number | null> {
  const ffprobe = runtimeBinary('ffmpeg').replace(/ffmpeg(\.exe)?$/i, 'ffprobe$1')
  const exe = fs.existsSync(ffprobe) ? ffprobe : null
  if (!exe) return null

  return new Promise((resolve) => {
    const child = spawn(
      exe,
      ['-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file],
      { windowsHide: true }
    )
    // Probing is a metadata read; if it has not answered in half a minute something is wrong
    // with the file and the frame sampler can fall back to a flat rate.
    const timer = setTimeout(() => child.kill('SIGKILL'), 30_000)
    let out = ''
    child.stdout?.on('data', (d: Buffer) => {
      out += d.toString()
    })
    // stderr is piped by default and nothing else reads it; an unread pipe that fills blocks
    // the child, which is exactly the hang this timeout would then have to clean up.
    child.stderr?.resume()
    child.on('close', () => {
      clearTimeout(timer)
      const n = Number(out.trim())
      resolve(Number.isFinite(n) && n > 0 ? n : null)
    })
    child.on('error', () => {
      clearTimeout(timer)
      resolve(null)
    })
  })
}

/**
 * Sample frames from a video with FFmpeg.
 *
 * llama.cpp can take the video directly, but doing the sampling ourselves gives control over
 * how many frames a given model sees, and it degrades gracefully on a build without video
 * support.
 */
export async function extractFrames(file: string, count: number): Promise<string[]> {
  const ffmpeg = runtimeBinary('ffmpeg')
  const duration = await videoDurationSeconds(file)
  const dir = path.join(TOOL_OUTPUT_DIR, `frames-${crypto.randomBytes(4).toString('hex')}`)
  await fsp.mkdir(dir, { recursive: true })

  // Spread frames across the clip rather than clustering at the start.
  const fps = duration && duration > 0 ? count / duration : 1
  const args = [
    '-hide_banner', '-loglevel', 'error',
    '-i', file,
    '-vf', `fps=${fps.toFixed(4)},scale='min(896,iw)':-1`,
    '-frames:v', String(count),
    path.join(dir, 'frame-%03d.jpg')
  ]

  await new Promise<void>((resolve, reject) => {
    const child = spawn(ffmpeg, args, { windowsHide: true })
    let stderr = ''
    let timedOut = false
    /*
     * Sampling a clip has to finish or fail; it cannot just sit there.
     *
     * This runs inside `chat:send` before the turn's abort controller is even registered, so an
     * ffmpeg that wedged on a malformed file produced a chat turn that could not be stopped by
     * any means short of killing the app.
     */
    const timer = setTimeout(() => {
      timedOut = true
      child.kill('SIGKILL')
    }, 5 * 60 * 1000)

    child.stderr?.on('data', (d: Buffer) => {
      stderr = `${stderr}${d.toString()}`.slice(-8192)
    })
    child.on('error', (err) => {
      clearTimeout(timer)
      reject(
        new Error(
          err.message.includes('ENOENT')
            ? 'FFmpeg is not bundled yet, so video cannot be processed. Run `npm run fetch-vendor`.'
            : err.message
        )
      )
    })
    child.on('close', (code) => {
      clearTimeout(timer)
      if (timedOut) return reject(new Error('FFmpeg took more than five minutes on this file and was stopped.'))
      return code === 0 ? resolve() : reject(new Error(stderr.slice(-500) || `ffmpeg exited ${code}`))
    })
  })

  const files = (await fsp.readdir(dir)).filter((f) => f.endsWith('.jpg')).sort()
  return files.map((f) => path.join(dir, f))
}

/**
 * Turn a user message plus attachments into multimodal content parts, refusing clearly
 * (rather than silently dropping) anything the loaded model cannot accept.
 */
export async function buildContent(
  text: string,
  attachments: string[],
  caps: ModelCapabilities
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
      const budget = frameBudget(caps)
      try {
        const frames = await extractFrames(file, budget.count)
        extractedFrames.push(...frames)
        for (const frame of frames) {
          parts.push({ type: 'image_url', image_url: { url: await toDataUrl(frame) } })
        }
        notes.push(
          `${path.basename(file)}: sampled ${frames.length} frames (${budget.reason}).` +
            (caps.nativeVideo ? '' : ' This model was not trained on video, so treat the result as a description of stills.')
        )
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
