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

/**
 * Pixels per frame for each preference, rather than a width.
 *
 * Budgeting by width overcharges anything that is not 16:9. A vertical phone video at a fixed
 * 640px wide is 640x1138 — nearly three times the area of the landscape frame the width implies,
 * and nearly three times the tokens, for a frame that carries no more information. Qwen's own
 * preprocessing budgets by total pixels for exactly this reason; these are the areas the old
 * widths worked out to at 16:9, so a landscape video is unchanged and everything else stops
 * being penalised for its shape.
 */
const DETAIL_PIXELS = { motion: 448 * 252, balanced: 640 * 360, detail: 896 * 504 } as const

/**
 * Bounds the encoder enforces on a single frame, in tokens.
 *
 * Below the floor a frame is scaled back up and the saving is imaginary; above the ceiling it is
 * scaled down and the extra pixels are discarded. Either way the budget would be spent on
 * something that does not happen.
 */
const MIN_TOKENS_PER_FRAME = 128
const MAX_TOKENS_PER_FRAME = 768

export type VideoDetail = keyof typeof DETAIL_PIXELS

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
  /*
   * Derive the frame's shape from an area budget and its aspect ratio, then round to the 28px
   * grid the encoder works in — a frame that is not a multiple of 28 is padded up to one, so
   * asking for 641 pixels of width costs exactly what 644 does.
   */
  const aspect = opts.aspect || 16 / 9
  const area = DETAIL_PIXELS[opts.detail] ?? DETAIL_PIXELS.balanced
  const grid = PIXELS_PER_TOKEN_EDGE
  const width = Math.max(grid, Math.round(Math.sqrt(area * aspect) / grid) * grid)
  const height = Math.max(grid, Math.round(width / aspect / grid) * grid)

  const perFrame = Math.min(
    MAX_TOKENS_PER_FRAME,
    Math.max(MIN_TOKENS_PER_FRAME, (width / grid) * (height / grid))
  )
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
  // Qwen's own preprocessing stops at 768 frames; beyond its tested range is not a good place
  // to be inventing behaviour.
  const MAX_FRAMES = 768
  if (opts.durationSeconds > 0) count = Math.min(count, Math.ceil(opts.durationSeconds * MAX_FPS))
  count = Math.max(1, Math.min(count, MAX_FRAMES))

  /*
   * An even number of frames, because they are consumed in pairs.
   *
   * The 3D convolution groups consecutive frames two at a time, so an odd count leaves the last
   * one unpaired and padded — it is charged for and contributes nothing. Rounding down rather
   * than up keeps the plan inside its budget.
   */
  if (count > 1) count -= count % 2

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
