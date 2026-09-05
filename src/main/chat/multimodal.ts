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
import os from 'node:os'
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
  /**
   * What became of each attachment, so the transcript can show it rather than name it.
   *
   * The sampled clip is the interesting half. Everything the model was actually shown of a video
   * — the frame rate, the crop, the resolution — is decided here and then described in a
   * sentence; keeping the artefact means the transcript can play it instead.
   */
  media: PreparedMedia[]
}

export interface PreparedMedia {
  /** The file as the user attached it. */
  source: string
  kind: 'image' | 'audio' | 'video' | 'doc'
  /** The re-encoded clip that was sent, when one was built. */
  optimised?: string
  /** Full-resolution stills sent alongside that clip. */
  stills?: string[]
  /** How the video was sampled, in the words the composer already uses. */
  note?: string
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

/**
 * How many threads ffmpeg may use, worked out from the machine it is running on.
 *
 * Worth being honest about what this buys: on the machine it was measured on, setting it made no
 * difference at all — 96.3s against 98.1s on the same file, which is noise. ffmpeg's default of
 * "auto" was already using what it needed, and the four-cores-of-twenty-four observation that
 * prompted this turned out to be the filter stage rather than the decoder.
 *
 * It stays because the default is a guess made by the build, not a guarantee, and an explicit
 * number derived from the machine cannot be worse than a conservative one chosen elsewhere.
 *
 * `availableParallelism` rather than `cpus().length`: it accounts for affinity masks and
 * container limits, so a machine that has been restricted to two cores is told two rather than
 * the number physically present. One core is kept back so the window still paints while a long
 * video is being prepared, and a two-core machine is given one thread rather than none.
 *
 * The ceiling is not arbitrary. Frame threads each hold their own reference frames, so past
 * roughly sixteen the decode stops getting faster while the memory footprint keeps climbing —
 * ffmpeg's own automatic choice caps in the same region for the same reason.
 */
const MAX_FFMPEG_THREADS = 16

function ffmpegThreads(): number {
  const raw = typeof os.availableParallelism === 'function' ? os.availableParallelism() : os.cpus()?.length
  const cores = Number.isFinite(raw) && (raw as number) > 0 ? (raw as number) : 1
  if (cores <= 2) return 1
  return Math.max(2, Math.min(cores - 1, MAX_FFMPEG_THREADS))
}

/**
 * Input-side options every pass shares.
 *
 * `-threads` has to precede `-i` to reach the decoder; placed after it, it configures the encoder
 * instead and the decode stays single-threaded-ish. That distinction is easy to get wrong and
 * silent when wrong, which is why it lives in one place.
 */
function decodeArgs(): string[] {
  return ['-threads', String(ffmpegThreads())]
}

/**
 * Decode only keyframes.
 *
 * `-discard`, not `-skip_frame`. The two read as synonyms and are not: `-skip_frame` asks the
 * decoder to skip frames it has already been handed, and dav1d simply ignores it — measured on a
 * 120 fps AV1 capture it returned all 18,169 frames and saved nothing at all. `-discard` drops
 * the packets before the decoder ever sees them, which no decoder can decline. The same file:
 *
 *     full decode at 2 fps    66.0s   303 frames
 *     -discard nokey           0.7s    73 frames
 *
 * Ninety-five times faster, still sampling across the whole clip, because keyframes are spread
 * through it by construction.
 *
 * Only for passes asking a coarse question — which part of the picture ever changes, roughly
 * where the content shifts. The sampling pass needs frames at particular moments rather than at
 * whatever moments the encoder happened to choose, so it pays for a real decode.
 */
const KEYFRAMES_ONLY = ['-discard', 'nokey']

/** Share of a video's budget spent on full-resolution stills rather than on the video itself. */
const KEYFRAME_SHARE = 0.3

/** However many cuts a video has, past this the stills stop earning their tokens. */
const MAX_KEYFRAMES = 24

/** How different a frame must be from the one before it to count as a new moment. */
const SCENE_THRESHOLD = 0.3

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
    const decodeGray = async (input: string[], filter: string): Promise<number> => {
      await runTool(
        exe,
        [
          '-hide_banner', '-y',
          ...input,
          ...decodeArgs(),
          '-i', file,
          '-vf', filter,
          '-fps_mode', 'passthrough',
          '-f', 'rawvideo', '-pix_fmt', 'gray',
          raw
        ],
        120_000
      )
      try {
        return Math.floor((await fsp.stat(raw)).size / (w * h))
      } catch {
        return 0
      }
    }

    /*
     * Keyframes first, because they are nearly free.
     *
     * Deciding which part of the picture ever moves does not need every frame — it needs frames
     * spread across the clip, and a keyframe-only decode gives exactly that while skipping the
     * 99% of frames that have to be reconstructed from their neighbours.
     */
    let frames = await decodeGray(KEYFRAMES_ONLY, `scale=${w}:${h},format=gray`)

    /*
     * Some encodes have almost no keyframes — a short clip, or one written as a single GOP.
     * Two frames is the minimum that can show a change at all, so below that fall back to
     * sampling by time and pay for the decode.
     */
    if (frames < 4) {
      frames = await decodeGray([], `fps=${REGION_PROBE_FPS},scale=${w}:${h},format=gray`)
    }

    const buf = await fsp.readFile(raw)
    const size = w * h
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
function pickSpread<T>(items: T[], n: number): T[] {
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

  const times = async (filter: string): Promise<number[]> => {
    const { stderr } = await runTool(
      exe,
      [
        '-hide_banner',
        ...KEYFRAMES_ONLY,
        ...decodeArgs(),
        '-i', file,
        // Downscaled first: the score only needs the gist, and comparing full-resolution frames
        // over a ten-minute video is far slower than the sampling it is meant to inform.
        '-vf', filter,
        '-fps_mode', 'passthrough',
        '-f', 'null',
        '-'
      ],
      120_000
    )
    const out: number[] = []
    for (const m of stderr.matchAll(/pts_time:([0-9.]+)/g)) {
      const t = Number(m[1])
      if (Number.isFinite(t)) out.push(t)
    }
    return out
  }

  /*
   * Both passes decode keyframes only, which is what makes this affordable.
   *
   * A 120 fps AV1 capture held 18,169 frames and 73 keyframes; asking the decoder for all of them
   * to find a couple of dozen interesting moments was most of the wait before a video was sent.
   *
   * What comes back is coarser than true frame-to-frame scene detection, and that is fine for
   * both callers: one spreads the moments across the clip and subdivides the gaps between them,
   * the other picks at most two dozen of them. Neither ranks by how strong the change was.
   */
  const cuts = await times(`scale=192:-2,select='gt(scene,${threshold})',showinfo`)
  if (cuts.length >= 4) return cuts

  /*
   * Too few, so take the keyframes themselves.
   *
   * Comparing keyframes seconds apart is not the comparison the threshold was chosen for, and on
   * gently changing footage it can select almost nothing. The keyframe positions are still a
   * reasonable set of candidate moments — encoders place them at cuts as well as periodically —
   * and a second keyframe-only pass costs a fraction of what one full decode would.
   */
  return times('scale=192:-2,showinfo')
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

/**
 * Sample the video and take its stills in a single decode.
 *
 * These were two passes over the same file, and on anything expensive to decode that was the
 * dominant cost of sending a video. The frames and the stills want different treatment — one is
 * reduced and cropped to carry motion, the other is full-size and whole to carry detail — but
 * they want it from the same pictures, so `split` feeds both from one decode and ffmpeg writes
 * two sets of files from one invocation.
 *
 * The stills branch does its own scene detection inside that shared decode, which is why nothing
 * has to go looking for the cuts separately afterwards.
 *
 * Returns more stills than will be used. Choosing which to keep is a spread across the clip, and
 * that cannot be expressed as a filter — `-frames:v` would take the first few and leave the end
 * of the video unrepresented — so the surplus is written and then deleted.
 */
async function extractSample(
  file: string,
  opts: {
    /** Sample at a fixed rate, for when the budget is dense enough to not need choosing. */
    fps?: number
    /** Or at these specific moments, when it is not. */
    times?: number[]
    sourceFps: number
    videoWidth: number
    /** Width for the full-resolution stills, or null to take none. */
    stillWidth: number | null
    sceneThreshold: number
    crop: LiveRegion | null
    /** Drop frames that are near-copies of the one before them. */
    decimate: boolean
  }
): Promise<{ frames: string[]; stills: string[] }> {
  const exe = runtimeBinary('ffmpeg')
  const base = path.join(TOOL_OUTPUT_DIR, `frames-${crypto.randomBytes(4).toString('hex')}`)
  const videoDir = path.join(base, 'video')
  const stillDir = path.join(base, 'stills')
  await fsp.mkdir(videoDir, { recursive: true })
  if (opts.stillWidth) await fsp.mkdir(stillDir, { recursive: true })

  /*
   * The tolerance is derived from the source rate so each requested moment matches exactly one
   * frame — too tight and moments are missed, too loose and neighbours are duplicated.
   */
  const selector = opts.times?.length
    ? `select='${opts.times
        .map((t) => `lt(abs(t-${t.toFixed(3)})\\,${(0.5 / Math.max(1, opts.sourceFps)).toFixed(4)})`)
        .join('+')}'`
    : `fps=${(opts.fps ?? 1).toFixed(5)}`

  // Crop before decimating and scaling: the duplicate test then compares only the part of the
  // picture that was ever going to matter, and the scaler is not asked to resample pixels that
  // are about to be discarded.
  const decimate = opts.decimate ? `${MPDECIMATE},` : ''
  const videoChain = `${selector},${cropFilter(opts.crop)}${decimate}scale='min(${opts.videoWidth},iw)':-2`

  const args = ['-hide_banner', '-y', ...decodeArgs(), '-i', file]

  if (opts.stillWidth) {
    args.push(
      '-filter_complex',
      `[0:v]split=2[a][b];` +
        `[a]${videoChain}[vid];` +
        // Deliberately uncropped: a caption or a chart can sit perfectly still and still be the
        // subject of the question, so the stills always carry the whole frame.
        `[b]select='gt(scene,${opts.sceneThreshold})',scale='min(${opts.stillWidth},iw)':-2[still]`,
      '-filter_complex_threads', String(ffmpegThreads()),
      '-map', '[vid]', '-fps_mode', 'passthrough', '-q:v', '3', path.join(videoDir, 'frame-%04d.jpg'),
      '-map', '[still]', '-fps_mode', 'passthrough', '-q:v', '3', path.join(stillDir, 'still-%04d.jpg')
    )
  } else {
    args.push(
      '-vf', videoChain,
      '-filter_threads', String(ffmpegThreads()),
      // `-fps_mode passthrough`, not `-vsync 0`: ffmpeg 9 removed `-vsync`, and an unrecognised
      // option is not a warning — the whole command fails and produces no frames at all.
      '-fps_mode', 'passthrough',
      '-q:v', '3',
      path.join(videoDir, 'frame-%04d.jpg')
    )
  }

  await runTool(exe, args, 600_000)

  const read = async (dir: string, prefix: string): Promise<string[]> => {
    try {
      return (await fsp.readdir(dir))
        .filter((f) => f.startsWith(prefix) && f.endsWith('.jpg'))
        .sort()
        .map((f) => path.join(dir, f))
    } catch {
      return []
    }
  }

  return { frames: await read(videoDir, 'frame-'), stills: await read(stillDir, 'still-') }
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
  opts: {
    contextLength: number
    share: number
    detail: VideoDetail
    temporalPairing: boolean
    trackScale?: number
    stillShare?: number
    maxFps?: number
    /** Drop frames that are near-copies of the one before them. */
    dropDuplicates?: boolean
    /** Crop away the part of the frame that never changes. */
    cropStatic?: boolean
    /**
     * Called as each pass begins.
     *
     * Preparing a video is several complete passes over the source, and on a long or awkwardly
     * encoded file that is minutes of silence. Naming the pass is the difference between a wait
     * and an apparent hang.
     */
    onStage?: (stage: string) => void
  }
): Promise<{
  frames: string[]
  condensed: string | null
  keyframes: string[]
  note: string
  /** What the model needs to be told about the clip's timeline, or null if there is nothing to say. */
  guidance: string | null
}> {
  const stage = opts.onStage ?? ((): void => {})
  stage('reading the video')
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
  if (probe && opts.cropStatic !== false) stage('finding what moves')
  const region = probe && opts.cropStatic !== false ? await liveRegion(file, probe) : null
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
    sourcePixels,
    trackScale: opts.trackScale,
    stillShare: opts.stillShare,
    maxFps: opts.maxFps
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
  // Sized by the planner now, so the budget is measured against what is actually sent rather
  // than against a full-size frame that then gets halved on the way out.
  const videoWidth = plan.trackWidth

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

  /*
   * How many stills the budget can carry, settled before the decode rather than after it.
   *
   * They used to be chosen once the video track was already extracted, which meant a second pass
   * over the source to go and fetch them. Knowing the number up front lets the same decode
   * produce both.
   */
  const maxStills = hybrid && duration > 0 ? plan.stills : 0
  const decimate = opts.dropDuplicates !== false
  const stillWidth = maxStills > 0 ? plan.width : null
  const sourceFps = probe?.fps ?? 30
  const uniformFps = duration > 0 ? Math.ceil(target) / duration : 1

  let sampled: { frames: string[]; stills: string[] }
  stage(`sampling ${plan.count} frames`)
  if (dense || duration <= 0) {
    sampled = await extractSample(file, {
      fps: uniformFps, sourceFps, videoWidth, stillWidth,
      sceneThreshold: SCENE_THRESHOLD, crop: region, decimate
    })
    how = 'evenly spaced'
  } else {
    const scenes = await detectScenes(file, SCENE_THRESHOLD)
    const times = chooseTimestamps(scenes, duration, Math.ceil(target))
    sampled = await extractSample(file, {
      ...(times.length ? { times } : { fps: uniformFps }),
      sourceFps, videoWidth, stillWidth, sceneThreshold: SCENE_THRESHOLD, crop: region, decimate
    })
    how = scenes.length ? `${scenes.length} scene changes, then the widest gaps` : 'evenly spaced'
  }
  frames = sampled.frames

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
  if (opts.temporalPairing && frames.length) stage('building the clip')
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
  if (hybrid && condensed && sampled.stills.length) {
    /*
     * Chosen from what the decode already produced.
     *
     * The stills branch writes one for every scene change it finds, which is usually more than
     * the budget can carry. Spread across the clip rather than taking the first few, because the
     * last minute of a video matters as much as the first — and the rest are deleted rather than
     * left behind, since nothing prunes the scratch directory.
     */
    keyframes = pickSpread(sampled.stills, maxStills)
    const keep = new Set(keyframes)
    await Promise.all(
      sampled.stills.filter((f) => !keep.has(f)).map((f) => fsp.rm(f, { force: true }).catch(() => undefined))
    )
  }

  const mins = Math.floor(duration / 60)
  const secs = Math.round(duration % 60)
  const actualFps = duration > 0 ? frames.length / duration : 0
  const dropped = Math.max(0, Math.ceil(target) - survived)
  const note =
    `sampled ${frames.length} frames from ${mins}m ${secs}s ` +
    `(${actualFps.toFixed(2)} fps at ${plan.trackWidth}px, ${how}` +
    (region ? `; cropped to the ${region.w}x${region.h} region that changes, ${Math.round((1 - region.share) * 100)}% of the frame never did` : '') +
    (condensed && duration > 0 && frames.length
      ? `; the clip plays ${(duration / (frames.length / SERVER_DECODE_FPS)).toFixed(1)}x faster than real time, which the model is told about`
      : '') +
    (dropped > 0 ? `; ${dropped} near-duplicate frames dropped` : '') +
    (keyframes.length ? `; plus ${keyframes.length} full-resolution stills at scene changes` : '') +
    `; about ${(frames.length * plan.costPerFrame + keyframes.length * plan.costPerStill).toLocaleString()} tokens)`

  /*
   * The clip's timeline is not the video's timeline, and the model has to be told.
   *
   * Frames are chosen across real time and then muxed at the rate the server decodes at, so a
   * clip covering two and a half minutes plays in seventy-five seconds. llama.cpp interleaves
   * its own timestamps, and those describe the clip, not the source — left unexplained the model
   * reasons about elapsed time and gets it wrong by exactly this factor. It showed up as a model
   * arguing with itself about whether an on-screen counter should advance half a second or a
   * whole one per frame.
   *
   * A sentence is cheap next to the frames it describes, and it is the only place this mapping
   * exists: nothing in the container records that the clip was sampled.
   */
  const clipSeconds = condensed && frames.length ? frames.length / SERVER_DECODE_FPS : 0
  const guidance =
    condensed && duration > 0 && clipSeconds > 0
      ? `The video that follows is a sample of a longer recording, not the recording itself. ` +
        `${frames.length} frames were taken across ${duration.toFixed(1)} seconds of source and are played back ` +
        `at ${SERVER_DECODE_FPS} fps, so the clip lasts ${clipSeconds.toFixed(1)} seconds and any timestamp you ` +
        `see for it must be multiplied by ${(duration / clipSeconds).toFixed(2)} to give the time in the original. ` +
        `Consecutive frames are ${(duration / frames.length).toFixed(2)} seconds apart in real time.`
      : null

  return { frames, condensed, keyframes, note, guidance }
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
  /** Ceiling on sampling rate, whatever the budget could otherwise afford. */
  maxFps?: number
  /** Size of the video track relative to a full frame. */
  trackScale?: number
  /** Share of the budget kept for full-resolution stills, when the track is reduced. */
  stillShare?: number
  /** Drop frames that are near-copies of the one before them. */
  dropDuplicates?: boolean
  /** Crop away the part of the frame that never changes. */
  cropStatic?: boolean
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
  video: VideoContext = DEFAULT_VIDEO,
  onStage?: (progress: { file: string; stage: string }) => void
): Promise<AttachmentPlan> {
  const parts: ContentPart[] = []
  const notes: string[] = []
  const extractedFrames: string[] = []
  /*
   * Only what was actually sent gets an entry.
   *
   * Every branch below can bail out with a note — no projector, no audio support, an unreadable
   * file — and a transcript that offered to play something the model never received would be
   * worse than one that said nothing.
   */
  const media: PreparedMedia[] = []

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
      media.push({ source: file, kind: 'image' })
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
      media.push({ source: file, kind: 'audio' })
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
          temporalPairing: video.serverTakesVideo,
          maxFps: video.maxFps,
          trackScale: video.trackScale,
          stillShare: video.stillShare,
          dropDuplicates: video.dropDuplicates,
          cropStatic: video.cropStatic,
          onStage: (stage) => onStage?.({ file: path.basename(file), stage })
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
          if (sampled.guidance) parts.push({ type: 'text', text: sampled.guidance })
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
          /*
           * The clip is kept, not just described.
           *
           * It already exists on disk and nothing deletes it, so recording where it went costs
           * nothing and turns "1.8 fps at 336px, cropped to the region that changes" from a
           * claim the user has to take on faith into something they can watch.
           */
          media.push({
            source: file,
            kind: 'video',
            optimised: sampled.condensed,
            stills: sampled.keyframes,
            note: sampled.note
          })
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
          // No clip to offer, but the stills are still what the model was shown.
          media.push({ source: file, kind: 'video', stills: sampled.frames, note: sampled.note })
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

  return { parts, notes, extractedFrames, media }
}
