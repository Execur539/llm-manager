/**
 * Where everything lives.
 *
 * Rule from the plan: the exe is portable and may be moved at any time, so persistent state
 * lives in %APPDATA%\LLMManager (which survives the move) while models live *beside the exe*
 * (which does not). The breadcrumb in APPDATA is what makes relocation possible at all.
 */

import { app } from 'electron'
import path from 'node:path'
import fs from 'node:fs'

/**
 * Root for all persistent state.
 *
 * LLMM_APPDATA_DIR redirects it, which end-to-end tests use to run against a throwaway
 * directory. Electron's own `appData` path ignores the APPDATA environment variable, so without
 * an explicit override a test would read and write the user's real settings, chat history and
 * relocation breadcrumb.
 */
export const APPDATA_DIR = process.env.LLMM_APPDATA_DIR
  ? path.resolve(process.env.LLMM_APPDATA_DIR)
  : path.join(app.getPath('appData'), 'LLMManager')
export const SETTINGS_FILE = path.join(APPDATA_DIR, 'settings.json')
export const SECRETS_FILE = path.join(APPDATA_DIR, 'secrets.json')
export const DB_FILE = path.join(APPDATA_DIR, 'llmmanager.db')
export const BREADCRUMB_FILE = path.join(APPDATA_DIR, 'models-path.json')
export const LOGS_DIR = path.join(APPDATA_DIR, 'logs')
export const SESSIONS_DIR = path.join(APPDATA_DIR, 'sessions')
export const CHECKPOINTS_DIR = path.join(APPDATA_DIR, 'checkpoints')
/** Full tool outputs that were truncated in-context but kept re-readable on disk. */
export const TOOL_OUTPUT_DIR = path.join(APPDATA_DIR, 'tool-output')

export const MODELS_DIR_NAME = 'LLMManagerModels'

/**
 * The directory the exe itself sits in.
 *
 * In development this is the project root; in production it is the folder the user put the
 * exe in, which is exactly what "models live beside the exe" needs.
 */
export function exeDir(): string {
  // Portable builds run from an extraction cache under LOCALAPPDATA, so `exe` points at the
  // unpacked copy rather than the file the user actually double-clicked. The launcher passes
  // the real location through, and it must win — otherwise "models live beside the exe" puts
  // them inside a cache directory that gets deleted on upgrade.
  const portable = process.env.LLMM_PORTABLE_DIR
  if (portable && fs.existsSync(portable)) return portable

  if (app.isPackaged) return path.dirname(app.getPath('exe'))
  return app.getAppPath()
}

/**
 * True when this process is running from the portable extraction cache.
 * Used to refuse writing user data anywhere underneath it.
 */
export function runtimeCacheDir(): string | null {
  const exe = path.dirname(app.getPath('exe'))
  return /[\\/]LLMManager[\\/]runtime-[^\\/]+$/i.test(exe) ? exe : null
}

/**
 * True when this process cannot know where the user's copy of the app actually is.
 *
 * The portable launcher unpacks to a cache directory and passes the real location through in
 * LLMM_PORTABLE_DIR. Run the unpacked copy directly — which is what happens if someone pins the
 * running app to the taskbar, since Windows resolves the pin to the executable it sees — and the
 * launcher never runs, the variable is never set, and `exeDir()` answers with the cache.
 *
 * Everything derived from that is then a guess: `defaultModelsDir()` falls back to Documents,
 * and anything comparing against it concludes the user's library is in the wrong place. A guess
 * is fine for choosing where to put a new folder. It is not fine as grounds for proposing to
 * move tens of gigabytes, so callers that would act on the exe's location check this first.
 */
export function exeLocationUnknown(): boolean {
  return runtimeCacheDir() !== null && !process.env.LLMM_PORTABLE_DIR
}

/** Would this path place user data inside the disposable extraction cache? */
export function isInsideRuntimeCache(target: string): boolean {
  const cache = runtimeCacheDir()
  if (!cache) return false
  const rel = path.relative(cache, path.resolve(target))
  return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel))
}

export function defaultModelsDir(): string {
  const beside = path.join(exeDir(), MODELS_DIR_NAME)
  // Last line of defence. If anything still resolves into the extraction cache, put models in
  // the user's Documents instead — a cache directory is deleted on upgrade, and losing a
  // multi-gigabyte library to a version bump is not a recoverable mistake.
  if (isInsideRuntimeCache(beside)) {
    return path.join(app.getPath('documents'), MODELS_DIR_NAME)
  }
  return beside
}

export function ensureDirs(): void {
  for (const dir of [APPDATA_DIR, LOGS_DIR, SESSIONS_DIR, CHECKPOINTS_DIR, TOOL_OUTPUT_DIR]) {
    fs.mkdirSync(dir, { recursive: true })
  }
}

export interface Breadcrumb {
  /** last known absolute path of the models directory */
  modelsDir: string
  /** the exe directory it was beside when we last saw it */
  exeDir: string
  updatedAt: number
}

export function readBreadcrumb(): Breadcrumb | null {
  try {
    return JSON.parse(fs.readFileSync(BREADCRUMB_FILE, 'utf8')) as Breadcrumb
  } catch {
    return null
  }
}

export function writeBreadcrumb(modelsDir: string): void {
  const crumb: Breadcrumb = { modelsDir, exeDir: exeDir(), updatedAt: Date.now() }
  fs.mkdirSync(APPDATA_DIR, { recursive: true })
  fs.writeFileSync(BREADCRUMB_FILE, JSON.stringify(crumb, null, 2))
}
