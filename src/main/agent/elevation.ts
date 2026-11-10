/**
 * Whether this process holds administrator rights.
 *
 * Told to the model up front. Left to find out for itself, it spent several tool calls per session
 * on it — and only after a command had already failed for the lack of them.
 */

import { spawnSync } from 'node:child_process'

let cached: boolean | null = null

export function isElevated(): boolean {
  if (cached !== null) return cached
  if (process.platform !== 'win32') {
    cached = process.getuid?.() === 0
    return cached
  }
  try {
    // Listing filter drivers needs administrator rights and nothing else, so its exit code is the answer.
    const probe = spawnSync('fltmc.exe', [], { windowsHide: true, stdio: 'ignore', timeout: 5000 })
    cached = probe.status === 0
  } catch {
    cached = false
  }
  return cached
}
