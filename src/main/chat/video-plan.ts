/**
 * How many frames a video gets, and how large each one is.
 *
 * This was a constant: sixteen frames for a video-trained model, six otherwise, however long the
 * clip. On a ten-minute video that is one frame every thirty-seven seconds — the limit was never
 * the context window, it was the number.
 *
 * Frames cost tokens by area. Qwen-family vision encoders cut a frame into 14x14 patches and
 * merge 2x2 of them into one token, so a token covers a 28x28 pixel block; a 3D convolution then
 * pairs consecutive frames, halving the count again when the video is sent as video rather than
 * as loose images. That makes the trade explicit and arithmetical: halve the width and a frame
 * costs a quarter as much, so four times as many fit.
 */
const PIXELS_PER_TOKEN_EDGE = 28

/**
 * What a frame costs as part of a video, relative to the same frame sent as an image.
 *
 * The 3D convolution that pairs consecutive frames suggests this should be a half, and it is
 * not. Measured against llama-server with Qwen3.8-27B, twelve 448x252 frames cost 1,505 prompt
 * tokens — about 125 each, against 144 for the naive patch count. So the video path is cheaper,
 * but by around a seventh rather than a half; the interleaved timestamps and the encoder's own
 * geometry account for the rest.
 *
 * Erring high is the safe direction. Under-estimating the cost means planning more frames than
 * fit and overflowing the window, which fails the request; over-estimating means a slightly
 * shorter sample than was strictly affordable, which nobody notices.
 */
const VIDEO_COST_FACTOR = 0.9

/** Frame widths for each preference. Heights follow the source aspect ratio. */
const DETAIL_WIDTHS = { motion: 448, balanced: 640, detail: 896 } as const

export type VideoDetail = keyof typeof DETAIL_WIDTHS

export interface VideoPlan {
  /** How many frames to sample. */
  count: number
  /** Width to scale them to; height follows the aspect ratio. */
  width: number
  /** Estimated tokens the whole video will occupy. */
  estimatedTokens: number
  /** Frames per second this works out to over the clip. */
  effectiveFps: number
}

/**
 * Work out a frame budget from the window the model actually has.
 *
 * `temporalPairing` halves the per-frame cost, and is only true when the frames will be sent as
 * a video — llama.cpp's 3D convolution pairs them. Sent as separate images they cost full price,
 * which is the single biggest reason to use the video path.
 */
export function planVideo(opts: {
  contextLength: number
  share: number
  detail: VideoDetail
  durationSeconds: number
  aspect: number
  temporalPairing: boolean
}): VideoPlan {
  const width = DETAIL_WIDTHS[opts.detail] ?? DETAIL_WIDTHS.balanced
  const height = Math.max(1, Math.round(width / (opts.aspect || 16 / 9)))
  const perFrame =
    Math.ceil(width / PIXELS_PER_TOKEN_EDGE) * Math.ceil(height / PIXELS_PER_TOKEN_EDGE)
  const costPerFrame = Math.max(1, Math.ceil(perFrame * (opts.temporalPairing ? VIDEO_COST_FACTOR : 1)))

  const budget = Math.max(0, Math.floor(opts.contextLength * opts.share))
  let count = Math.floor(budget / costPerFrame)

  /*
   * Two ceilings that have nothing to do with the budget.
   *
   * There is no value in sampling faster than the source changes, and Qwen3-VL is trained and
   * evaluated at 2-4 fps — beyond that the extra frames are near-duplicates paying full price.
   * The absolute cap is a guard against a pathological duration turning into tens of thousands
   * of ffmpeg seeks.
   */
  const MAX_FPS = 4
  const MAX_FRAMES = 2048
  if (opts.durationSeconds > 0) count = Math.min(count, Math.ceil(opts.durationSeconds * MAX_FPS))
  count = Math.max(1, Math.min(count, MAX_FRAMES))

  return {
    count,
    width,
    estimatedTokens: count * costPerFrame,
    effectiveFps: opts.durationSeconds > 0 ? count / opts.durationSeconds : 0
  }
}

/**
 * Choose which moments to sample, preferring where the picture actually changes.
 *
 * Uniform sampling spends the budget evenly whether or not anything is happening, which on a
 * lecture or a screen recording means most frames are the same slide. ffmpeg's scene score is a
 * cheap measure of how much a frame differs from the one before it, so the cuts it reports are
 * where the information is.
 *
 * Scenes alone are not enough either: a video with three cuts would get three frames, losing
 * everything that happens *within* a long take. So scene changes are taken first, and the rest
 * of the budget fills the largest remaining gaps — coverage is never sacrificed for salience.
 */
export function chooseTimestamps(scenes: number[], duration: number, count: number): number[] {
  if (duration <= 0 || count <= 0) return []
  const picked: number[] = []

  // Always anchor the start; a video whose first cut is at 40s should still show its opening.
  picked.push(0)
  for (const t of scenes) {
    if (picked.length >= count) break
    if (t > 0 && t < duration) picked.push(t)
  }

  /*
   * Fill what is left by repeatedly halving the widest gap.
   *
   * Simpler than it sounds and better than padding uniformly: it puts each remaining frame where
   * the current sample is thinnest, so a long unbroken take gets subdivided while a densely cut
   * sequence is left alone.
   */
  while (picked.length < count) {
    picked.sort((a, b) => a - b)
    let widest = 0
    let at = -1
    for (let i = 0; i < picked.length; i++) {
      const next = i + 1 < picked.length ? picked[i + 1] : duration
      const gap = next - picked[i]
      if (gap > widest) {
        widest = gap
        at = i
      }
    }
    if (at === -1 || widest <= 0.001) break
    const next = at + 1 < picked.length ? picked[at + 1] : duration
    picked.push(picked[at] + (next - picked[at]) / 2)
  }

  return picked.sort((a, b) => a - b).slice(0, count)
}
