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
    idealContext: 131072,
    headroomMb: 768
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
    contextShare: 0.45
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
  { path: ['video', 'contextShare'], min: 0.05, max: 0.8 }
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
