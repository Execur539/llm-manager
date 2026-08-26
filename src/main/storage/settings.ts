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

let cache: AppSettings | null = null

export function loadSettings(): AppSettings {
  if (cache) return cache
  try {
    const raw = JSON.parse(fs.readFileSync(SETTINGS_FILE, 'utf8'))
    cache = merge(DEFAULT_SETTINGS, raw)
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
  const next = merge(loadSettings(), patch)
  saveSettings(next)
  return next
}
