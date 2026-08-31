/**
 * Locating bundled binaries.
 *
 * In production these live inside the extracted runtime cache beside the exe; in development
 * they are expected under ./vendor. Everything resolves through here so there is exactly one
 * place that knows the layout, and a missing binary produces a clear error instead of a
 * confusing spawn failure deep in a tool.
 */

import { app } from 'electron'
import path from 'node:path'
import fs from 'node:fs'
import type { Backend } from '@shared/types'

export type BinaryName = 'llama-server' | 'ffmpeg' | 'cloudflared' | 'python' | 'node' | 'rg'

function vendorRoot(): string {
  // Escape hatch for development and test harnesses, which may run with a different app path.
  if (process.env.LLMM_VENDOR_DIR) return process.env.LLMM_VENDOR_DIR

  // Packaged: electron-builder copies `extraResources` into the resources directory, so the
  // vendor tree sits at <resources>/vendor. This must not be derived from the exe path —
  // the portable build's exe lives in an extraction cache and LLMM_PORTABLE_DIR deliberately
  // points elsewhere, so anything computed from `exe` would look in the wrong place.
  if (app.isPackaged) return path.join(process.resourcesPath, 'vendor')

  // Dev: <project>/vendor
  return path.join(app.getAppPath(), 'vendor')
}

const BACKEND_DIRS: Record<Backend, string> = {
  cuda: 'llama.cpp/cuda',
  vulkan: 'llama.cpp/vulkan',
  cpu: 'llama.cpp/cpu'
}

export function llamaServerPath(backend: Backend): string {
  return path.join(vendorRoot(), BACKEND_DIRS[backend], 'llama-server.exe')
}

/**
 * Resolve a bundled binary, falling back to the system PATH.
 * Node in particular is always available because Electron ships it as its own executable.
 */
export function runtimeBinary(name: BinaryName): string {
  if (name === 'node') {
    // Reuse Electron's bundled Node by running the app binary in Node mode.
    return process.execPath
  }

  const candidates = [
    path.join(vendorRoot(), name, `${name}.exe`),
    path.join(vendorRoot(), `${name}.exe`),
    path.join(vendorRoot(), name, 'bin', `${name}.exe`)
  ]
  if (name === 'python') {
    candidates.unshift(path.join(vendorRoot(), 'python', 'python.exe'))
  }

  for (const c of candidates) {
    if (fs.existsSync(c)) return c
  }
  // Fall back to PATH so development works before vendor binaries are fetched.
  return `${name}.exe`
}

/**
 * FFmpeg must be reachable on PATH for llama.cpp's video frame extraction, which shells out
 * to it. We inject our bundled copy's directory rather than relying on the user having it.
 */
export function childEnv(): NodeJS.ProcessEnv {
  const ffmpeg = runtimeBinary('ffmpeg')
  /*
   * Only ever prepend a real, absolute directory.
   *
   * `runtimeBinary` falls back to a bare `ffmpeg.exe` when the vendor tree has not been fetched,
   * and `path.dirname('ffmpeg.exe')` is `"."` — which `existsSync` happily confirms, because the
   * working directory always exists. Every child spawned in that state therefore ran with the
   * current directory at the *front* of its PATH, which is the oldest trick in the book for
   * getting the wrong executable picked up. When ffmpeg is missing the right answer is to change
   * nothing and let the tool that needs it report that it is missing.
   */
  const dir = path.dirname(ffmpeg)
  if (!path.isAbsolute(dir) || !fs.existsSync(dir)) return { ...process.env }

  return {
    ...process.env,
    PATH: `${dir}${path.delimiter}${process.env.PATH ?? ''}`
  }
}

/**
 * The bundled Chromium used for browser automation.
 * Returns null when it has not been fetched, so the browser tools can fail with a useful
 * message instead of a Playwright launch error.
 */
export function chromiumExecutable(): string | null {
  const root = path.join(vendorRoot(), 'chromium')
  const candidates = [
    path.join(root, 'chrome.exe'),
    path.join(root, 'chrome-win', 'chrome.exe'),
    path.join(root, 'chrome-win64', 'chrome.exe')
  ]
  for (const c of candidates) {
    if (fs.existsSync(c)) return c
  }
  return null
}

export function vendorDir(): string {
  return vendorRoot()
}

/**
 * Diagnostic detail for the "setup incomplete" panel.
 *
 * Reports the directory that was searched, because a wrong vendor root looks exactly like
 * missing downloads — which is precisely how this went unnoticed in the first packaged build.
 */
export function vendorDiagnostics(backend: Backend): {
  root: string
  rootExists: boolean
  missing: string[]
  present: string[]
} {
  const root = vendorRoot()
  const missing = missingBinaries(backend)
  const present = ['llama.cpp', 'ffmpeg', 'python', 'cloudflared', 'rg', 'chromium', 'models'].filter((d) =>
    fs.existsSync(path.join(root, d))
  )
  return { root, rootExists: fs.existsSync(root), missing, present }
}

export function missingBinaries(backend: Backend): string[] {
  const missing: string[] = []
  if (!fs.existsSync(llamaServerPath(backend))) missing.push(`llama-server (${backend})`)
  for (const n of ['ffmpeg', 'python', 'cloudflared', 'rg'] as BinaryName[]) {
    const p = runtimeBinary(n)
    if (!path.isAbsolute(p)) missing.push(n)
  }
  if (!chromiumExecutable()) missing.push('chromium')
  if (!fs.existsSync(embeddingModelPath())) missing.push('embedding model')
  return missing
}

/** The small embedding model bundled for RAG; users may point at another in Settings. */
export function embeddingModelPath(): string {
  return path.join(vendorRoot(), 'models', 'embedding.gguf')
}
