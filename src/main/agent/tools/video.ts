/**
 * Watching a video, a piece at a time.
 *
 * A video is far too large to show a model whole, and deciding in advance which frames matter
 * means guessing what the question will be. So the model looks for itself: a handful of frames
 * spread across the whole thing to learn what is in it, then closer and sharper looks at whichever
 * moments turn out to matter. Each look costs only the frames it asked for.
 *
 * Frames go to the model as separate pictures, each with the time it was taken from. They are not
 * packed into a clip for the server's video path: that path pairs neighbouring frames and reads
 * each pair as one, which is right for frames a fraction of a second apart and smears two
 * unrelated pictures together when they are half a minute apart, as an overview's are.
 */

import fs from 'node:fs'
import path from 'node:path'
import { spawn } from 'node:child_process'
import type { Tool, ToolOutput } from './base'
import { schema, str, int } from './base'
import type { ContentPart } from '../../runtime/llama'
import { runtimeBinary } from '../../runtime/binaries'

const num = (description: string): Record<string, unknown> => ({ type: 'number', description })

/** Frame width and the most frames one look may ask for, per level of detail. */
const DETAIL = {
  low: { width: 448, maxFrames: 24, what: 'enough to tell what is happening' },
  medium: { width: 784, maxFrames: 16, what: 'enough to make out faces, objects and large text' },
  high: { width: 1288, maxFrames: 8, what: 'enough to read small text' }
} as const
type Detail = keyof typeof DETAIL

/** The vision encoder's grid: a frame costs about one token per 28 pixel square. */
const PATCH = 28

export interface VideoFacts {
  duration: number
  width: number
  height: number
  fps: number
  hasSound: boolean
}

function run(exe: string, args: string[], timeoutMs: number, signal?: AbortSignal): Promise<{ stdout: string; stderr: string; code: number | null }> {
  return new Promise((resolve) => {
    const child = spawn(exe, args, { windowsHide: true })
    let stdout = ''
    let stderr = ''
    const timer = setTimeout(() => child.kill('SIGKILL'), timeoutMs)
    const stop = (): void => {
      child.kill('SIGKILL')
    }
    signal?.addEventListener('abort', stop, { once: true })
    child.stdout?.on('data', (d: Buffer) => (stdout += d.toString()))
    child.stderr?.on('data', (d: Buffer) => (stderr += d.toString()))
    const done = (code: number | null): void => {
      clearTimeout(timer)
      signal?.removeEventListener('abort', stop)
      resolve({ stdout, stderr, code })
    }
    child.on('close', done)
    child.on('error', () => done(null))
  })
}

/** The same, keeping what ffmpeg writes to stdout as bytes: a frame piped out rather than saved. */
function runBytes(exe: string, args: string[], timeoutMs: number, signal?: AbortSignal): Promise<Buffer> {
  return new Promise((resolve) => {
    const child = spawn(exe, args, { windowsHide: true })
    const chunks: Buffer[] = []
    const timer = setTimeout(() => child.kill('SIGKILL'), timeoutMs)
    const stop = (): void => {
      child.kill('SIGKILL')
    }
    signal?.addEventListener('abort', stop, { once: true })
    child.stdout?.on('data', (d: Buffer) => chunks.push(d))
    child.stderr?.resume()
    const done = (): void => {
      clearTimeout(timer)
      signal?.removeEventListener('abort', stop)
      resolve(Buffer.concat(chunks))
    }
    child.on('close', done)
    child.on('error', done)
  })
}

function ffprobe(): string {
  return runtimeBinary('ffmpeg').replace(/ffmpeg(\.exe)?$/i, 'ffprobe$1')
}

/**
 * Length, shape, frame rate and whether there is a soundtrack. Null when it is not a readable video.
 *
 * `file` is a path or a URL: the chat scratchpad serves a video it holds in memory over loopback
 * HTTP, which ffprobe and ffmpeg read with range requests as if it were a file.
 */
export async function videoFacts(file: string, signal?: AbortSignal): Promise<VideoFacts | null> {
  const { stdout } = await run(
    ffprobe(),
    ['-v', 'error', '-show_entries', 'format=duration:stream=codec_type,width,height,avg_frame_rate', '-of', 'json', file],
    30_000,
    signal
  )
  try {
    const j = JSON.parse(stdout) as {
      format?: { duration?: string }
      streams?: { codec_type?: string; width?: number; height?: number; avg_frame_rate?: string }[]
    }
    const video = j.streams?.find((s) => s.codec_type === 'video')
    const duration = Number(j.format?.duration ?? 0)
    if (!video || !video.width || !video.height || !Number.isFinite(duration) || duration <= 0) return null
    const [n, d] = (video.avg_frame_rate ?? '0/1').split('/').map(Number)
    return {
      duration,
      width: video.width,
      height: video.height,
      fps: d > 0 && n > 0 ? n / d : 30,
      hasSound: !!j.streams?.some((s) => s.codec_type === 'audio')
    }
  } catch {
    return null
  }
}

/** Seconds as m:ss, with tenths when the moment is not on a whole second. */
export function clock(seconds: number): string {
  const s = Math.max(0, seconds)
  const whole = Math.floor(s)
  const tenths = Math.round((s - whole) * 10)
  const carry = tenths === 10 ? 1 : 0
  const total = whole + carry
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const sec = total % 60
  const tail = `${String(sec).padStart(2, '0')}${tenths > 0 && tenths < 10 ? `.${tenths}` : ''}`
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${tail}` : `${m}:${tail}`
}

/**
 * The moments to look at: `count` of them, evenly spread over a stretch of the video.
 *
 * Taken from the middle of equal slices rather than from their edges, so an overview does not
 * spend its first frame on a fade-in and its last on the credits, and so a single frame asked of
 * a stretch lands in the middle of it. Never more frames than the stretch actually contains.
 */
export function pickMoments(start: number, end: number, count: number, fps: number, duration: number): number[] {
  const from = Math.min(Math.max(0, start), duration)
  const to = Math.min(Math.max(from, end), duration)
  const span = to - from
  // A hair of tolerance: 10.1 - 10 is not quite a tenth of a second in floating point.
  const available = Math.max(1, Math.floor(span * fps + 1e-6))
  const n = Math.max(1, Math.min(count, available))
  // The very end of a file has no frame to seek to; stay one frame short of it.
  const last = Math.max(0, duration - 1 / fps)
  const moments = Array.from({ length: n }, (_, i) => Math.min(last, from + ((i + 0.5) * span) / n))
  return span === 0 ? [Math.min(last, from)] : moments
}

/**
 * One frame at one moment, as a JPEG no wider than `width`, or null when none could be read.
 *
 * Piped out on stdout rather than written to a file, so looking at a video leaves nothing on disk:
 * that matters for the chat scratchpad, which keeps what it holds in memory unless told otherwise.
 */
async function frameAt(source: string, at: number, width: number, signal?: AbortSignal): Promise<Buffer | null> {
  const bytes = await runBytes(
    runtimeBinary('ffmpeg'),
    [
      '-hide_banner', '-loglevel', 'error',
      // Before the input: seeks by index to the nearest keyframe and decodes only from there, so
      // a frame late in a long video costs the same as one at the start.
      '-ss', at.toFixed(3),
      '-i', source,
      '-frames:v', '1',
      '-vf', `scale='min(${width},iw)':-2`,
      '-q:v', '2',
      '-f', 'image2pipe',
      '-c:v', 'mjpeg',
      'pipe:1'
    ],
    60_000,
    signal
  )
  // A JPEG starts FF D8; anything else is an error message or nothing at all.
  return bytes.length > 2 && bytes[0] === 0xff && bytes[1] === 0xd8 ? bytes : null
}

/** What an attached or named video is, in a sentence, for a model that has not looked at it yet. */
export function describeVideo(name: string, facts: VideoFacts): string {
  return (
    `${path.basename(name)} is ${clock(facts.duration)} long, ${facts.width}x${facts.height} at ` +
    `${Math.round(facts.fps)} frames a second, ${facts.hasSound ? 'with sound' : 'without sound'}`
  )
}

/** What one look at a video asks for, as the model wrote it. */
export interface LookRequest {
  start?: unknown
  end?: unknown
  frames?: unknown
  detail?: unknown
}

/**
 * Look at part of a video: frames spread over the stretch asked for, each labelled with its time.
 *
 * Shared by the agent's watch_video, which names a file on disk, and chat's, which names something
 * in its scratchpad. `source` is what ffmpeg reads (a path, or a loopback URL); `name` is what the
 * model calls it, so the text it gets back never mentions a URL or a folder it cannot use.
 */
export async function lookAtVideo(
  source: string,
  name: string,
  args: LookRequest,
  opts: { vision: boolean; signal: AbortSignal; again?: string }
): Promise<ToolOutput> {
  const facts = await videoFacts(source, opts.signal)
  if (!facts) return `${name} could not be read as a video.`
  const about = describeVideo(name, facts)
  if (!opts.vision) {
    return `${about}. The loaded model has no vision projector, so its frames cannot be shown to you.`
  }

  const detail: Detail = (['low', 'medium', 'high'] as const).includes(String(args.detail) as Detail)
    ? (String(args.detail) as Detail)
    : 'low'
  const level = DETAIL[detail]
  const start = Number.isFinite(Number(args.start)) ? Number(args.start) : 0
  const end = args.end === undefined || args.end === null || !Number.isFinite(Number(args.end)) ? facts.duration : Number(args.end)
  if (start >= facts.duration) {
    return `${about}. It ends at ${clock(facts.duration)}, so there is nothing at ${clock(start)}.`
  }
  const wanted = Math.max(1, Math.round(Number(args.frames ?? 8)) || 8)
  const moments = pickMoments(start, end, Math.min(wanted, level.maxFrames), facts.fps, facts.duration)

  // A few at a time: each is its own short decode, and a dozen at once would only contend.
  const frames: (Buffer | null)[] = new Array(moments.length).fill(null)
  let next = 0
  await Promise.all(
    Array.from({ length: Math.min(4, moments.length) }, async () => {
      while (next < moments.length && !opts.signal.aborted) {
        const i = next++
        frames[i] = await frameAt(source, moments[i], level.width, opts.signal)
      }
    })
  )

  const media: ContentPart[] = []
  const shown: number[] = []
  for (let i = 0; i < moments.length; i++) {
    const bytes = frames[i]
    if (!bytes) continue
    shown.push(moments[i])
    media.push({ type: 'text', text: `Frame at ${clock(moments[i])}:` })
    media.push({ type: 'image_url', image_url: { url: `data:image/jpeg;base64,${bytes.toString('base64')}` } })
  }
  if (!shown.length) return `${about}. No frames could be read between ${clock(start)} and ${clock(end)}.`

  const shownWidth = Math.min(level.width, facts.width)
  // Even, as the scaler makes it.
  const shownHeight = Math.round((shownWidth * facts.height) / facts.width / 2) * 2
  const tokensEach = Math.ceil(shownWidth / PATCH) * Math.ceil(shownHeight / PATCH)
  const gap = shown.length > 1 ? (shown[shown.length - 1] - shown[0]) / (shown.length - 1) : 0
  const capped = wanted > level.maxFrames ? ` (${level.maxFrames} is the most at ${detail} detail)` : ''

  const text =
    `${about}.\n` +
    `Showing ${shown.length} frame${shown.length === 1 ? '' : 's'}${capped} from ${clock(shown[0])} to ` +
    `${clock(shown[shown.length - 1])}` +
    (gap > 0 ? `, about ${gap >= 1 ? `${gap.toFixed(gap >= 10 ? 0 : 1)} s` : `${Math.round(gap * 1000)} ms`} apart` : '') +
    `, at ${detail} detail (${shownWidth}x${shownHeight}, ${level.what}; roughly ${(tokensEach * shown.length).toLocaleString()} tokens).\n` +
    `Times: ${shown.map(clock).join(', ')}.\n` +
    (opts.again ?? 'To see a moment more closely, call watch_video again with start and end around it, and more frames or higher detail.')
  return { text, media }
}

const watchVideo: Tool = {
  name: 'watch_video',
  description:
    'Look at a video file. With no times it shows frames spread across the whole video, which is ' +
    'how to find out what is in it. Give start and end, in seconds, to look at one part, and more ' +
    'frames or higher detail to see it more closely. Every frame is labelled with its time. Look ' +
    'as many times as you need: a cheap overview first, then closer looks at the moments that ' +
    'matter, is far cheaper than asking for everything at once. You see pictures only, not sound.',
  tier: 'read',
  parameters: schema(
    {
      path: str('Path to the video file'),
      start: num('Where to start looking, in seconds from the beginning (default: the beginning)'),
      end: num('Where to stop looking, in seconds (default: the end of the video)'),
      frames: int('How many frames to look at across that stretch (default 8)'),
      detail: str(
        "'low' (default; small frames, cheap), 'medium' (faces, objects, large text) or 'high' (small text; few frames at a time)"
      )
    },
    ['path']
  ),
  async run(args, ctx): Promise<ToolOutput> {
    const file = path.isAbsolute(String(args.path))
      ? path.normalize(String(args.path))
      : path.resolve(ctx.cwd, String(args.path))
    if (!fs.existsSync(file)) return `${file} does not exist.`
    return lookAtVideo(file, file, args, { vision: ctx.vision, signal: ctx.signal })
  }
}

export const videoTools: Tool[] = [watchVideo]
