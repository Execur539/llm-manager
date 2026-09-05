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
 * The 3D convolution that pairs consecutive frames says this should be a half, and it is. An
 * earlier measurement here put it at 0.9 and reasoned about why the saving was smaller than the
 * architecture implied — that reasoning was explaining an artefact. The clip it measured was
 * encoded at half the rate the server decodes at, so every frame was being sent twice; the
 * constant was fitted to the duplication rather than to the encoder.
 *
 * Re-measured against llama-server with Qwen3.8-27B once the encode rate matched, across three
 * frame sizes and two frame counts:
 *
 *     448x252, 12 frames    809 tokens    67.4/frame   vs 144 naive    0.468
 *     644x364, 12 frames  1,565 tokens   130.4/frame   vs 299 naive    0.436
 *     336x196, 12 frames    487 tokens    40.6/frame   vs  84 naive    0.483
 *     448x252, 24 frames  1,502 tokens    62.6/frame   vs 144 naive    0.435
 *
 * Mean 0.456 with a spread of 0.049, so the constant is rounded up rather than to the mean.
 * Erring high is the safe direction: under-estimating the cost means planning more frames than
 * fit and overflowing the window, which fails the request; over-estimating means a slightly
 * shorter sample than was strictly affordable, which nobody notices.
 */
const VIDEO_COST_FACTOR = 0.5

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
const DETAIL_PIXELS = {
  motion: 448 * 252,
  balanced: 640 * 360,
  detail: 896 * 504,
  /*
   * 720p, for footage where the answer is written on the screen.
   *
   * Expensive and honest about it: a frame this size costs roughly four times a balanced one, so
   * the frame rate drops in proportion. Worth it for a screen recording or anything where small
   * text decides the answer, wasteful for footage where the motion is the point.
   */
  high: 1280 * 720
} as const

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
  /** Full frame width, used for the stills; height follows the aspect ratio. */
  width: number
  /** Width of the video track itself, which may be reduced below the full frame. */
  trackWidth: number
  /** How many full-resolution stills the budget leaves room for. */
  stills: number
  /** What one track frame and one still each cost, so an actual result can be priced. */
  costPerFrame: number
  costPerStill: number
  /** Estimated tokens the whole video will occupy, track and stills together. */
  estimatedTokens: number
  /** Frames per second this works out to over the clip. */
  effectiveFps: number
}

/** However many cuts a video has, past this the stills stop earning their tokens. */
export const MAX_STILLS = 24

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
  /**
   * Pixels actually available in a source frame, after any crop.
   *
   * Upscaling buys nothing: the extraction clamps to the source width, so a frame smaller than
   * the budget costs less than the budget assumed. Without this the planner believes it spent
   * the whole allowance, under-counts what is left, and buys fewer frames than it can afford —
   * which matters most for exactly the clips that were cropped hardest.
   */
  sourcePixels?: number
  /**
   * Fraction of the full frame the video track is sent at, between a quarter and all of it.
   *
   * Below 1 the track trades legibility for frame rate and leaves room for full-size stills at
   * the cuts; at 1 it carries the detail itself and the stills are dropped as redundant.
   */
  trackScale?: number
  /** Share of the budget reserved for those stills. */
  stillShare?: number
  /** Ceiling on sampling rate, whatever the budget could otherwise afford. */
  maxFps?: number
}): VideoPlan {
  /*
   * Derive the frame's shape from an area budget and its aspect ratio, then round to the 28px
   * grid the encoder works in — a frame that is not a multiple of 28 is padded up to one, so
   * asking for 641 pixels of width costs exactly what 644 does.
   */
  const aspect = opts.aspect || 16 / 9
  const budgetArea = DETAIL_PIXELS[opts.detail] ?? DETAIL_PIXELS.balanced
  const available = opts.sourcePixels && opts.sourcePixels > 0 ? opts.sourcePixels : Number.POSITIVE_INFINITY
  const area = Math.min(budgetArea, available)
  const grid = PIXELS_PER_TOKEN_EDGE

  /*
   * Rounded down to the grid rather than to the nearest when the source is what limits us.
   *
   * Rounding to the nearest can overshoot by up to a full cell — a 240px source plans as 252 —
   * and the extraction clamps to the source width regardless, so the plan would be charging for
   * a column of pixels that never arrives. Down is also the safe direction for the budget.
   */
  const sourceBound = available < budgetArea
  const snap = sourceBound
    ? (n: number): number => Math.floor(n / grid) * grid
    : (n: number): number => Math.round(n / grid) * grid
  const width = Math.max(grid, snap(Math.sqrt(area * aspect)))
  const height = Math.max(grid, snap(width / aspect))

  /*
   * The video track is a fraction of the full frame, and the planner has to know it.
   *
   * It used to be halved after the fact, downstream of the budget, so the plan believed it was
   * spending on 644px frames while 336px ones were sent — and once the per-frame cost was
   * corrected downward, that halving left four fifths of the allowance unspent. Sizing the track
   * here means the budget is measured against what is actually sent.
   */
  const trackWidth = Math.max(grid, snap(width * clamp(opts.trackScale ?? 1, 0.25, 1)))
  const trackHeight = Math.max(grid, snap(trackWidth / aspect))

  const tokensFor = (w: number, h: number): number =>
    Math.min(MAX_TOKENS_PER_FRAME, Math.max(MIN_TOKENS_PER_FRAME, (w / grid) * (h / grid)))

  const pairing = opts.temporalPairing ? VIDEO_COST_FACTOR : 1
  const costPerFrame = Math.max(1, Math.ceil(tokensFor(trackWidth, trackHeight) * pairing))
  // Stills are sent as images, so they never get the pairing discount.
  const costPerStill = Math.max(1, Math.ceil(tokensFor(width, height)))

  const budget = Math.max(0, Math.floor(opts.contextLength * opts.share))

  /*
   * Stills are only worth reserving for when the track is reduced enough to pay for them.
   *
   * At full resolution the track already carries the detail they exist to restore, and spending
   * a third of the window on near-duplicates of frames that are already legible is waste.
   */
  const stillShare = opts.temporalPairing && trackWidth < width ? clamp(opts.stillShare ?? 0.3, 0, 0.6) : 0
  const stills = Math.min(MAX_STILLS, Math.floor((budget * stillShare) / costPerStill))
  const trackBudget = Math.max(0, budget - stills * costPerStill)

  let count = Math.floor(trackBudget / costPerFrame)

  /*
   * Two ceilings that have nothing to do with the budget.
   *
   * The frame rate cap matters more than it used to. While frames were being charged twice, the
   * budget ran out long before this ceiling did, so its value barely mattered; at the real price
   * a two-minute clip can afford roughly six hundred frames, and without a ceiling the planner
   * would buy every one of them and spend the whole saving on frame rate nobody asked for.
   *
   * Two is the knee rather than an arbitrary retreat from four. Qwen3-VL is trained and
   * evaluated at 2-4 fps, so this stays inside its range, and the literature on frame sampling
   * puts the practical default at 1-2 fps with returns flattening above that — beyond it the
   * extra frames are near-duplicates paying full price, which is precisely what mpdecimate then
   * has to throw away again. Raising it back to 4 is a one-line change for anyone who wants to
   * spend the budget that way.
   *
   * The absolute cap is a guard against a pathological duration turning into tens of thousands
   * of ffmpeg seeks.
   */
  const maxFps = clamp(opts.maxFps ?? 2, 0.1, 4)
  // Qwen's own preprocessing stops at 768 frames; beyond its tested range is not a good place
  // to be inventing behaviour.
  const MAX_FRAMES = 768
  if (opts.durationSeconds > 0) count = Math.min(count, Math.ceil(opts.durationSeconds * maxFps))
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
    trackWidth,
    stills,
    costPerFrame,
    costPerStill,
    estimatedTokens: count * costPerFrame + stills * costPerStill,
    effectiveFps: opts.durationSeconds > 0 ? count / opts.durationSeconds : 0
  }
}

function clamp(n: number, lo: number, hi: number): number {
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : lo
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
