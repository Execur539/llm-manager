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
  // Packaged: <exe dir>/.llmmanager-runtime/vendor  (written once on first run)
  // Dev:      <project>/vendor
  if (app.isPackaged) return path.join(path.dirname(app.getPath('exe')), '.llmmanager-runtime', 'vendor')
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
  const dir = path.dirname(ffmpeg)
  const sep = path.delimiter
  return {
    ...process.env,
    PATH: fs.existsSync(dir) ? `${dir}${sep}${process.env.PATH ?? ''}` : process.env.PATH
  }
}

export function missingBinaries(backend: Backend): string[] {
  const missing: string[] = []
  if (!fs.existsSync(llamaServerPath(backend))) missing.push(`llama-server (${backend})`)
  for (const n of ['ffmpeg', 'python'] as BinaryName[]) {
    const p = runtimeBinary(n)
    if (!path.isAbsolute(p)) missing.push(n)
  }
  return missing
}
