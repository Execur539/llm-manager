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

export const APPDATA_DIR = path.join(app.getPath('appData'), 'LLMManager')
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
  if (app.isPackaged) return path.dirname(app.getPath('exe'))
  return app.getAppPath()
}

export function defaultModelsDir(): string {
  return path.join(exeDir(), MODELS_DIR_NAME)
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
