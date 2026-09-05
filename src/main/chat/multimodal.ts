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
  /** Source dimensions in pixels, so a crop rectangle can be expressed in them. */
  width: number
  height: number
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

/**
 * Drop frames that are near-copies of the one before them.
 *
 * The single biggest saving available, because the cost of a frame has nothing to do with how
 * much it tells you. A screen recording or a talking head holds still for long stretches, and
 * those frames are charged in full for repeating what the previous one already said — a two
 * minute capture spent fifty thousand tokens largely on frames that were nearly identical.
 *
 * Applied after sampling, so it compares the frames actually being sent rather than the source's
 * own. The thresholds are ffmpeg's defaults: they drop what is visually indistinguishable and
 * keep anything with real movement in it, which is the behaviour wanted here — a video that
 * genuinely changes throughout loses nothing at all.
 */
const MPDECIMATE = 'mpdecimate=hi=64*12:lo=64*5:frac=0.33'

/** The encoder's patch grid; frame dimensions are rounded to it, so choosing off it is waste. */
const PATCH_GRID = 28

/**
 * The rate llama.cpp decodes video at, which the condensed clip has to be muxed at.
 *
 * Not a preference — a fixed property of the server. Newer builds expose `--video-fps`, but this
 * one has only `--media-path`, so the rate is hardcoded upstream and the clip must meet it. Mux
 * below it and every frame is decoded twice and charged twice; mux above it and frames we paid
 * to select are thrown away before the encoder ever sees them.
 *
 * If the bundled llama.cpp is ever updated, check `--help` for `--video-fps` and either pass it
 * explicitly or re-confirm this default before trusting it.
 */
const SERVER_DECODE_FPS = 4

/** Share of a video's budget spent on full-resolution stills rather than on the video itself. */
const KEYFRAME_SHARE = 0.3

/** However many cuts a video has, past this the stills stop earning their tokens. */
const MAX_KEYFRAMES = 24

/**
 * Width of the pass that decides which part of the picture is worth paying for.
 *
 * Deliberately tiny. The question is only "does this region ever change", which survives heavy
 * downscaling, and a 160px pass over a ten-minute video costs a fraction of the extraction it
 * informs.
 */
const REGION_PROBE_WIDTH = 160

/** Sampling rate for that pass. Two per second is ample to tell movement from stillness. */
const REGION_PROBE_FPS = 2

/** Below this much variation across the whole clip, a pixel is codec noise rather than motion. */
const REGION_NOISE = 8

/**
 * How much of the frame the live region must save before it is worth cropping to.
 *
 * Cropping is not free of risk: something static can still be informative, and a region that
 * covers most of the frame anyway buys little for that risk. Requiring a fifth of the frame back
 * keeps the crop to cases where it clearly pays — screen recordings, fixed-camera footage,
 * letterboxed uploads — and leaves handheld or full-frame footage completely alone.
 */
const REGION_MIN_SAVING = 0.8

/** A rectangle of source pixels, and what fraction of the frame it covers. */
interface LiveRegion {
  w: number
  h: number
  x: number
  y: number
  share: number
}

/**
 * Find the rectangle the picture actually uses.
 *
 * Two different kinds of waste have the same shape and the same cure. Letterbox and pillarbox
 * bars are charged for at exactly the same rate as content — a phone video padded into a 16:9
 * container measured 68% dead pixels, all of them black, all of them paid for in every frame.
 * And a screen recording or a fixed-camera talking head spends most of its area on a desktop or
 * a wall that is identical in every single frame. Both are regions that never change.
 *
 * So rather than detecting bars specifically, this measures change: decode a thumbnail-sized
 * grayscale pass, track each pixel's high and low water mark across the whole clip, and take the
 * bounding box of everything whose range clears the noise floor. Bars have zero range and fall
 * outside it; so does the static half of a screen recording.
 *
 * ffmpeg's own `cropdetect` was the obvious tool and is the wrong one — it scans inward from the
 * edges looking for a border, so interior motion gives it nonsense (it returned a negative height
 * when tried). Doing the measurement directly is both simpler and correct.
 *
 * Measured on synthetic cases: a 1920x1080 screen recording with one active pane came back at 9%
 * of the frame, a fixed-camera talking head at 10%, and handheld footage that genuinely fills the
 * frame at 100% — which is the important one, because it means the technique declines to act
 * rather than damaging content that needs the whole frame.
 *
 * Returns null when there is nothing worth cropping, which callers treat as "use the whole frame".
 */
async function liveRegion(file: string, probe: VideoProbe): Promise<LiveRegion | null> {
  const exe = runtimeBinary('ffmpeg')
  if (!fs.existsSync(exe) || !probe.width || !probe.height) return null

  const w = REGION_PROBE_WIDTH
  const h = Math.max(2, Math.round((probe.height / probe.width) * w / 2) * 2)
  const raw = path.join(TOOL_OUTPUT_DIR, `region-${crypto.randomBytes(4).toString('hex')}.gray`)

  try {
    /*
     * Written to a file rather than read from stdout: this is binary and `runTool` hands back
     * decoded strings, which would corrupt every byte above 0x7f.
     */
    await runTool(
      exe,
      [
        '-hide_banner', '-y',
        '-i', file,
        '-vf', `fps=${REGION_PROBE_FPS},scale=${w}:${h},format=gray`,
        '-f', 'rawvideo', '-pix_fmt', 'gray',
        raw
      ],
      120_000
    )

    const buf = await fsp.readFile(raw)
    const size = w * h
    const frames = Math.floor(buf.length / size)
    // One frame cannot show change, so there is nothing to conclude.
    if (frames < 2) return null

    const hi = Buffer.alloc(size, 0)
    const lo = Buffer.alloc(size, 255)
    for (let f = 0; f < frames; f++) {
      const off = f * size
      for (let i = 0; i < size; i++) {
        const v = buf[off + i]
        if (v > hi[i]) hi[i] = v
        if (v < lo[i]) lo[i] = v
      }
    }

    let x1 = w
    let x2 = -1
    let y1 = h
    let y2 = -1
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x
        if (hi[i] - lo[i] <= REGION_NOISE) continue
        if (x < x1) x1 = x
        if (x > x2) x2 = x
        if (y < y1) y1 = y
        if (y > y2) y2 = y
      }
    }
    // A clip where nothing moves at all: a still frame, or a probe that failed to decode.
    if (x2 < 0 || y2 < 0) return null

    /*
     * A margin, because the probe is coarse and the cost of cutting into the subject is far
     * higher than the cost of carrying a little dead space around it.
     */
    const pad = 2
    x1 = Math.max(0, x1 - pad)
    y1 = Math.max(0, y1 - pad)
    x2 = Math.min(w - 1, x2 + pad)
    y2 = Math.min(h - 1, y2 + pad)

    const sx = probe.width / w
    const sy = probe.height / h
    // Even values throughout: yuv420p subsamples chroma and rejects odd dimensions and offsets.
    const even = (n: number): number => Math.max(2, Math.round(n / 2) * 2)
    const cw = Math.min(probe.width, even((x2 - x1 + 1) * sx))
    const ch = Math.min(probe.height, even((y2 - y1 + 1) * sy))
    const cx = Math.min(probe.width - cw, even(x1 * sx))
    const cy = Math.min(probe.height - ch, even(y1 * sy))

    const share = (cw * ch) / (probe.width * probe.height)
    if (share > REGION_MIN_SAVING) return null
    return { w: cw, h: ch, x: cx, y: cy, share }
  } catch {
    // Nothing here is load-bearing; a failed probe just means the whole frame is used.
    return null
  } finally {
    await fsp.rm(raw, { force: true }).catch(() => {})
  }
}

/** Take `n` items spread across a list, rather than the first `n`. */
function pickSpread(items: number[], n: number): number[] {
  if (n <= 0 || !items.length) return []
  if (items.length <= n) return items
  const step = items.length / n
  return Array.from({ length: n }, (_, i) => items[Math.floor(i * step)])
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
    const width = s?.width ?? 0
    const height = s?.height ?? 0
    const aspect = width && height ? width / height : 16 / 9
    if (!Number.isFinite(duration) || duration <= 0) return null
    return { duration, fps, aspect, width, height }
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
async function extractAt(
  file: string,
  times: number[],
  width: number,
  crop?: LiveRegion | null
): Promise<string[]> {
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
      // Crop before decimating and scaling: dropping the dead area first means the duplicate
      // test compares only the part of the picture that was ever going to matter, and the
      // scaler is not asked to resample pixels about to be discarded.
      '-vf', `select='${expr}',${cropFilter(crop)}${MPDECIMATE},scale='min(${width},iw)':-2`,
      /*
       * Keep every selected frame rather than re-timing them to a constant rate.
       *
       * `-fps_mode passthrough`, not `-vsync 0`: ffmpeg 9 removed `-vsync`, and an unrecognised
       * option is not a warning — the whole command fails and produces no frames at all. This
       * path only runs when the sample is sparse, which is why it went unnoticed.
       */
      '-fps_mode', 'passthrough',
      '-q:v', '3',
      path.join(dir, 'frame-%04d.jpg')
    ],
    300_000
  )
  const files = (await fsp.readdir(dir)).filter((f) => f.endsWith('.jpg')).sort()
  return files.map((f) => path.join(dir, f))
}

/** A leading `crop=` stage for the filter chain, or nothing when the whole frame is in use. */
function cropFilter(crop?: LiveRegion | null): string {
  return crop ? `crop=${crop.w}:${crop.h}:${crop.x}:${crop.y},` : ''
}

/** Evenly spaced frames, for when the sample is dense enough that scene detection adds nothing. */
async function extractUniform(
  file: string,
  count: number,
  duration: number,
  width: number,
  crop?: LiveRegion | null
): Promise<string[]> {
  const exe = runtimeBinary('ffmpeg')
  const dir = path.join(TOOL_OUTPUT_DIR, `frames-${crypto.randomBytes(4).toString('hex')}`)
  await fsp.mkdir(dir, { recursive: true })
  const fps = duration > 0 ? count / duration : 1
  await runTool(
    exe,
    [
      '-hide_banner',
      '-i', file,
      '-vf', `fps=${fps.toFixed(5)},${cropFilter(crop)}${MPDECIMATE},scale='min(${width},iw)':-2`,
      // No -frames:v ceiling: decimation decides how many survive, and capping here would cut
      // the tail of the video rather than its redundancy. `-fps_mode` because ffmpeg 9 removed
      // `-vsync`, and an unrecognised option fails the command outright.
      '-fps_mode', 'passthrough',
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
  /*
   * One output frame per frame we chose, and no more.
   *
   * This clip was encoded at 2 fps while the server decodes video at SERVER_DECODE_FPS, so
   * llama.cpp was resampling upward and handing the vision encoder each frame twice. Measured
   * against Qwen3.8-27B: the same twelve pictures cost 1,616 prompt tokens muxed at 2 fps and
   * 809 muxed at 4 — exactly double, for identical content.
   *
   * The rate is not cosmetic metadata, it is the contract with the decoder. Frame duration is
   * derived from it rather than written separately so the two cannot drift apart.
   */
  const dur = (1 / SERVER_DECODE_FPS).toFixed(4)
  // Concat demuxer rather than a numbered pattern: the frames are already named in order and
  // this avoids re-deriving a glob that has to match exactly.
  await fsp.writeFile(
    listFile,
    frames.map((f) => `file '${f.replace(/\\/g, '/')}'\nduration ${dur}`).join('\n') +
      `\nfile '${frames[frames.length - 1].replace(/\\/g, '/')}'\n`,
    'utf8'
  )
  const out = path.join(dir, 'condensed.mp4')
  await runTool(
    exe,
    ['-hide_banner', '-y', '-f', 'concat', '-safe', '0', '-i', listFile,
     '-r', String(SERVER_DECODE_FPS), '-pix_fmt', 'yuv420p', out],
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
): Promise<{ frames: string[]; condensed: string | null; keyframes: string[]; note: string }> {
  const probe = await probeVideo(file)
  const duration = probe?.duration ?? 0

  /*
   * Which part of the frame is worth paying for, decided before the budget is spent.
   *
   * This has to come first because it changes the shape being budgeted for: a phone video
   * padded into a 16:9 container is planned as 16:9 and charged for bars, where the picture
   * inside it is 9:16. Planning against the live rectangle spends the area budget on the
   * subject instead.
   */
  const region = probe ? await liveRegion(file, probe) : null
  const aspect = region ? region.w / region.h : (probe?.aspect ?? 16 / 9)
  const sourcePixels = region
    ? region.w * region.h
    : probe?.width && probe?.height
      ? probe.width * probe.height
      : undefined

  const plan = planVideo({
    contextLength: opts.contextLength,
    share: opts.share,
    detail: opts.detail,
    durationSeconds: duration,
    aspect,
    temporalPairing: opts.temporalPairing,
    sourcePixels
  })

  /*
   * Scene detection earns its cost only when the sample is sparse.
   *
   * Above roughly one frame a second the uniform sample already lands on every cut, and a second
   * full decode of the video to prove it is pure overhead. Below that, the cuts are exactly what
   * a uniform sample misses.
   */
  /*
   * Detail is not spread evenly, because information is not.
   *
   * Frames at a cut carry something new; the frames between them mostly carry continuity, and
   * paying full resolution for continuity is where the budget goes. So the video is sent at half
   * width — a quarter of the pixels, a quarter of the tokens — to hold motion, order and
   * timestamps, and a handful of full-resolution stills are sent alongside it for the moments
   * that actually changed. Twenty per cent of frames at full size and eighty at a quarter costs
   * forty per cent of sending everything at full size.
   *
   * Only when the server can take video: as separate images there is no cheap carrier for the
   * motion, and the split would just be a worse sample.
   */
  const hybrid = opts.temporalPairing
  const videoWidth = hybrid ? Math.max(PATCH_GRID, Math.round((plan.width * 0.5) / PATCH_GRID) * PATCH_GRID) : plan.width

  const dense = duration > 0 && plan.count / duration >= 1
  let frames: string[]
  let how: string
  let keyframes: string[] = []

  /*
   * Sample beyond the budget and let decimation bring it back.
   *
   * Asking for exactly the budget and then dropping duplicates would simply spend less, which is
   * half the point. Oversampling first means the frames that survive are spread more finely over
   * the parts of the video that actually change, so the saving is turned into temporal
   * resolution where it is useful rather than banked everywhere.
   */
  const OVERSAMPLE = 1.8
  const target = Math.min(plan.count * OVERSAMPLE, duration > 0 ? duration * 4 : plan.count * OVERSAMPLE)

  if (dense || duration <= 0) {
    frames = await extractUniform(file, Math.ceil(target), duration, videoWidth, region)
    how = 'evenly spaced'
  } else {
    const scenes = await detectScenes(file)
    const times = chooseTimestamps(scenes, duration, Math.ceil(target))
    frames = times.length
      ? await extractAt(file, times, videoWidth, region)
      : await extractUniform(file, Math.ceil(target), duration, videoWidth, region)
    how = scenes.length ? `${scenes.length} scene changes, then the widest gaps` : 'evenly spaced'
  }

  /*
   * Whatever survived decimation, trimmed to what the budget can pay for.
   *
   * Dropped evenly rather than from the end, so a video that is busy throughout keeps its
   * coverage instead of stopping partway through.
   */
  const survived = frames.length
  if (frames.length > plan.count) {
    const step = frames.length / plan.count
    frames = Array.from({ length: plan.count }, (_, i) => frames[Math.floor(i * step)])
  }
  // Consumed in pairs, so an odd frame at the end is charged for and contributes nothing.
  if (frames.length > 1 && frames.length % 2 === 1) frames = frames.slice(0, -1)

  let condensed: string | null = null
  if (opts.temporalPairing && frames.length) {
    condensed = await encodeCondensed(frames, path.dirname(frames[0]))
  }

  /*
   * Full-resolution stills for the moments that changed.
   *
   * Bounded by what the reduced video freed up rather than by a fixed count, so the pair always
   * costs less than sending everything at full size. Spread across the cuts rather than taking
   * the first few, because the last minute of a video matters as much as the first.
   */
  if (hybrid && condensed && duration > 0) {
    const stillCost = Math.max(1, Math.round(plan.estimatedTokens / Math.max(1, plan.count)))
    const affordable = Math.floor((plan.estimatedTokens * KEYFRAME_SHARE) / stillCost)
    if (affordable >= 1) {
      const cuts = await detectScenes(file)
      const chosen = pickSpread(cuts.filter((t) => t > 0 && t < duration), Math.min(affordable, MAX_KEYFRAMES))
      /*
       * Deliberately uncropped, unlike the video track.
       *
       * Cropping to the live region is safe for the video because the video's job is motion, and
       * by definition nothing outside that region moves. It is not safe as the *only* view of the
       * clip: a burnt-in caption, a title card or a chart can sit perfectly still and still be
       * the thing being asked about. Sending the stills whole means the model always has the full
       * frame somewhere, and the aggressive crop costs nothing it cannot recover.
       */
      if (chosen.length) keyframes = await extractAt(file, chosen, plan.width)
    }
  }

  const mins = Math.floor(duration / 60)
  const secs = Math.round(duration % 60)
  const perFrame = plan.count > 0 ? plan.estimatedTokens / plan.count : 0
  const actualFps = duration > 0 ? frames.length / duration : 0
  const dropped = Math.max(0, Math.ceil(target) - survived)
  const note =
    `sampled ${frames.length} frames from ${mins}m ${secs}s ` +
    `(${actualFps.toFixed(2)} fps at ${plan.width}px, ${how}` +
    (region ? `; cropped to the ${region.w}x${region.h} region that changes, ${Math.round((1 - region.share) * 100)}% of the frame never did` : '') +
    (dropped > 0 ? `; ${dropped} near-duplicate frames dropped` : '') +
    (keyframes.length ? `; plus ${keyframes.length} full-resolution stills at scene changes` : '') +
    `; about ${Math.round(frames.length * perFrame * (hybrid ? 0.25 : 1) + keyframes.length * perFrame).toLocaleString()} tokens)`

  return { frames, condensed, keyframes, note }
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
  /** Set when the projector failed to allocate at load, so media would crash the server. */
  visionUnavailable?: string
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

    /*
     * Refused rather than sent, when the projector never got its memory at load time.
     *
     * Sending this would not give a poor answer, it would segfault the server and take the
     * loaded model and the conversation down with it. Checked before the branches rather than
     * inside each of them, so an image and a video cannot drift apart on it.
     */
    if ((kind === 'image' || kind === 'video') && video.visionUnavailable) {
      notes.push(`${path.basename(file)} skipped: ${video.visionUnavailable}`)
      continue
    }

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

          /*
           * The stills follow the video, so the model reads them as detail on what it just saw.
           *
           * The video is at reduced resolution and carries motion, order and timing; these carry
           * what the moments actually looked like. Sent after rather than before, because a
           * still ahead of the clip reads as a separate subject instead of a close-up of it.
           */
          for (const still of sampled.keyframes) {
            parts.push({ type: 'image_url', image_url: { url: await toDataUrl(still) } })
          }
          extractedFrames.push(...sampled.keyframes)
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
