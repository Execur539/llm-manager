/**
 * Settings store. Plain JSON in %APPDATA% — small, inspectable, and trivially backed up.
 * Reads are merged over defaults so a settings file written by an older build never
 * leaves a new field undefined.
 */

import fs from 'node:fs'
import path from 'node:path'
import type { AppSettings } from '@shared/types'
import { APPDATA_DIR, SETTINGS_FILE } from './paths'

export const DEFAULT_SETTINGS: AppSettings = {
  modelsDir: null,
  hfToken: null,
  autoFit: {
    minKvType: 'q4_0',
    preferredKvType: 'q8_0',
    targetContext: 65536,
    /*
     * As much as the model was trained for, rather than a fixed 131,072.
     *
     * This was the real ceiling on every load: a model trained to 262,144 was planned at half its
     * context however much VRAM was free, because the ideal never asked for more. The planner
     * caps at the trained length on its own, so a large number here means "as much as the model
     * and the card allow" rather than "as much as possible at any cost" — it will not trade cache
     * precision for length.
     */
    idealContext: 1048576,
    headroomMb: 768,
    /*
     * Plan past the trained length using rope scaling. Off deliberately.
     *
     * It works, and llama.cpp applies the scaling at every length rather than only past the
     * trained one, so short prompts are affected too. Long-context evaluations also find most
     * models degrading before their advertised limit, so extending past a *trained* 262,144 is
     * buying something the model was already struggling to deliver.
     */
    allowRopeScaling: false
  },
  agent: {
    enabled: true,
    planMode: false,
    compaction: 'auto-compact',
    maxToolCallsPerTurn: 50,
    commandTimeoutMs: 120000,
    hardBlocksDisabled: false,
    remoteToolsEnabled: false
  },
  ultra: {
    // Three is the smallest count where a majority can form and the synthesis pass has
    // something to compare rather than merely arbitrate.
    samples: 3,
    thinkingFactor: 2.5,
    maxContinuations: 4
  },
  downloads: {
    connections: 4
  },
  video: {
    /*
     * How to spend the context window a video is allowed.
     *
     * Frames cost tokens by area, so this is a straight trade: smaller frames buy more of them.
     * 'motion' favours temporal resolution — many small frames, for following what happens;
     * 'detail' favours legibility — fewer large frames, for reading text on screen. The frame
     * count is derived from the model's actual window rather than fixed, so a larger context
     * automatically buys a longer video rather than the same sixteen frames.
     */
    detail: 'balanced',
    /** Share of the context window a single video may occupy. */
    contextShare: 0.45,
    /**
     * Ceiling on how often the video is sampled, whatever the budget could otherwise afford.
     *
     * Qwen3-VL is trained and evaluated between two and four frames a second, and the sampling
     * literature puts the practical default at one to two. Past that the extra frames are
     * near-duplicates paying full price.
     */
    maxFps: 2,
    /**
     * How large the video track is relative to a full frame.
     *
     * Below 1 the clip is sent reduced and full-size stills are sent alongside it for the moments
     * that changed; at 1 the clip carries the detail itself and the stills are dropped as
     * redundant. Reducing it buys frame rate and costs legibility.
     */
    trackScale: 1,
    /** Share of the budget reserved for those stills, when the track is reduced. */
    stillShare: 0.3,
    /** Drop frames that are near-copies of the one before them. */
    dropDuplicates: true,
    /** Crop away the part of the frame that never changes across the clip. */
    cropStatic: true
  },
  server: {
    enabled: false,
    port: 1234,
    apiKey: null,
    jitLoad: true
  },
  remote: {
    enabled: false,
    mode: 'tunnel',
    domain: null
  },
  ui: {
    closeAction: 'ask'
  }
}

/** Deep-merge stored settings over defaults so missing keys are filled, not undefined. */
function merge<T>(base: T, override: unknown): T {
  if (override === null || typeof override !== 'object' || Array.isArray(override)) return base
  const out = { ...base } as Record<string, unknown>
  for (const [k, v] of Object.entries(override as Record<string, unknown>)) {
    const b = (base as Record<string, unknown>)[k]
    if (b !== null && typeof b === 'object' && !Array.isArray(b)) out[k] = merge(b, v)
    else if (v !== undefined) out[k] = v
  }
  return out as T
}

/**
 * Numeric settings that other code does arithmetic or comparisons on, with the range each is
 * only ever allowed to hold.
 *
 * The settings file is plain JSON that a person can edit, and until recently the UI wrote to it
 * on every keystroke through `Number(input.value)` — so an empty box stored 0 and a lone minus
 * sign stored NaN, which `JSON.stringify` writes as `null`. `maxToolCallsPerTurn` is the one
 * that bites: the agent loop is `while (calls < max)`, and both `0 < 0` and `0 < null` are
 * false, so the agent answered nothing at all and said nothing about why.
 *
 * Clamped on read rather than only on write, because a file that already holds a bad value has
 * to heal itself — a fix in the UI does nothing for the install that already broke.
 */
const NUMERIC_BOUNDS: { path: [keyof AppSettings, string]; min: number; max: number }[] = [
  { path: ['autoFit', 'targetContext'], min: 512, max: 10_000_000 },
  { path: ['autoFit', 'idealContext'], min: 512, max: 10_000_000 },
  { path: ['autoFit', 'headroomMb'], min: 0, max: 65_536 },
  { path: ['agent', 'maxToolCallsPerTurn'], min: 1, max: 1000 },
  { path: ['agent', 'commandTimeoutMs'], min: 1000, max: 3_600_000 },
  { path: ['ultra', 'samples'], min: 1, max: 8 },
  { path: ['ultra', 'maxContinuations'], min: 0, max: 20 },
  { path: ['server', 'port'], min: 1, max: 65_535 },
  { path: ['downloads', 'connections'], min: 1, max: 16 },
  // A video may not take the whole window: the question about it has to fit too.
  { path: ['video', 'contextShare'], min: 0.05, max: 0.8 },
  // Below a quarter frame per second a two-minute clip is eight pictures; above four the model
  // is outside the range it was trained on.
  { path: ['video', 'maxFps'], min: 0.25, max: 4 },
  // A quarter-size track is already a sixteenth of the pixels; smaller is not legible.
  { path: ['video', 'trackScale'], min: 0.25, max: 1 },
  { path: ['video', 'stillShare'], min: 0, max: 0.6 }
]

function clampNumerics(settings: AppSettings): AppSettings {
  const store = settings as unknown as Record<string, Record<string, unknown>>
  const defaults = DEFAULT_SETTINGS as unknown as Record<string, Record<string, unknown>>

  for (const { path: [section, key], min, max } of NUMERIC_BOUNDS) {
    const value = store[section as string]?.[key]
    if (typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max) continue
    const fallback =
      typeof value === 'number' && Number.isFinite(value)
        ? Math.min(max, Math.max(min, Math.round(value)))
        : defaults[section as string][key]
    if (store[section as string]) store[section as string][key] = fallback
  }
  return settings
}

let cache: AppSettings | null = null

export function loadSettings(): AppSettings {
  if (cache) return cache
  try {
    const raw = JSON.parse(fs.readFileSync(SETTINGS_FILE, 'utf8'))
    cache = clampNumerics(merge(DEFAULT_SETTINGS, raw))
  } catch {
    cache = { ...DEFAULT_SETTINGS }
  }
  return cache
}

export function saveSettings(next: AppSettings): void {
  cache = next
  fs.mkdirSync(APPDATA_DIR, { recursive: true })
  const tmp = path.join(APPDATA_DIR, `.settings.${process.pid}.tmp`)
  fs.writeFileSync(tmp, JSON.stringify(next, null, 2))
  fs.renameSync(tmp, SETTINGS_FILE) // atomic-ish; avoids a truncated file on crash
}

export function patchSettings(patch: Partial<AppSettings>): AppSettings {
  // Clamped on the way in as well as on the way out, so a caller that is not the settings UI —
  // the API server, a remote session, a hand-edited file reloaded — cannot store a value the
  // rest of the app will trip over.
  const next = clampNumerics(merge(loadSettings(), patch))
  saveSettings(next)
  return next
}
