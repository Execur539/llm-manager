/**
 * Ask-first updater.
 *
 * Checks a release feed, shows what changed, and applies only on the user's say-so — no silent
 * background updates, by decision. The llama.cpp backends ship inside the app bundle, so an app
 * update is a backend update.
 *
 * Applying an update swaps the exe on next launch via a small batch script, because Windows
 * will not let a running executable overwrite itself.
 */

import { app } from 'electron'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'
import { spawn } from 'node:child_process'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { logger } from '../log'

/** Where releases are published. Overridable so a fork can point elsewhere. */
const UPDATE_FEED =
  process.env.LLMM_UPDATE_FEED ?? 'https://api.github.com/repos/llm-manager/llm-manager/releases/latest'

export interface UpdateInfo {
  available: boolean
  currentVersion: string
  latestVersion: string | null
  notes: string | null
  downloadUrl: string | null
  bytes: number
  publishedAt: string | null
  error?: string
}

function compareVersions(a: string, b: string): number {
  const pa = a.replace(/^v/, '').split('.').map(Number)
  const pb = b.replace(/^v/, '').split('.').map(Number)
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const x = pa[i] ?? 0
    const y = pb[i] ?? 0
    if (x !== y) return x - y
  }
  return 0
}

export async function checkForUpdate(): Promise<UpdateInfo> {
  const currentVersion = app.getVersion()
  try {
    const res = await fetch(UPDATE_FEED, {
      headers: { Accept: 'application/vnd.github+json', 'User-Agent': 'llm-manager' },
      signal: AbortSignal.timeout(15000)
    })
    if (!res.ok) {
      return {
        available: false,
        currentVersion,
        latestVersion: null,
        notes: null,
        downloadUrl: null,
        bytes: 0,
        publishedAt: null,
        error: `Update feed returned HTTP ${res.status}`
      }
    }

    const release = (await res.json()) as {
      tag_name: string
      body?: string
      published_at?: string
      assets: { name: string; browser_download_url: string; size: number }[]
    }

    const asset = release.assets.find((a) => /\.exe$/i.test(a.name))
    const latestVersion = release.tag_name.replace(/^v/, '')

    return {
      available: compareVersions(latestVersion, currentVersion) > 0 && !!asset,
      currentVersion,
      latestVersion,
      notes: release.body ?? null,
      downloadUrl: asset?.browser_download_url ?? null,
      bytes: asset?.size ?? 0,
      publishedAt: release.published_at ?? null
    }
  } catch (err) {
    return {
      available: false,
      currentVersion,
      latestVersion: null,
      notes: null,
      downloadUrl: null,
      bytes: 0,
      publishedAt: null,
      error: err instanceof Error ? err.message : String(err)
    }
  }
}

/**
 * Download the new exe next to the current one, then leave a batch script that swaps them
 * after this process exits and relaunches.
 */
export async function applyUpdate(
  url: string,
  onProgress: (p: { done: number; total: number }) => void
): Promise<{ ok: boolean; message: string }> {
  if (!app.isPackaged) {
    return { ok: false, message: 'Updates only apply to a packaged build. In development, pull and rebuild.' }
  }

  const currentExe = app.getPath('exe')
  const dir = path.dirname(currentExe)
  const staged = path.join(dir, `.update-${Date.now()}.exe`)

  const res = await fetch(url, { redirect: 'follow' })
  if (!res.ok || !res.body) throw new Error(`Download failed: HTTP ${res.status}`)

  const total = Number(res.headers.get('content-length') ?? 0)
  let done = 0
  const body = Readable.fromWeb(res.body as Parameters<typeof Readable.fromWeb>[0])
  body.on('data', (chunk: Buffer) => {
    done += chunk.length
    onProgress({ done, total })
  })
  await pipeline(body, fs.createWriteStream(staged))

  if (total > 0 && done !== total) {
    await fsp.rm(staged, { force: true })
    throw new Error(`Download was truncated (${done} of ${total} bytes). Nothing was changed.`)
  }

  // Windows will not let a running exe overwrite itself, so a helper does it after exit.
  const script = path.join(os.tmpdir(), `llmm-update-${Date.now()}.cmd`)
  await fsp.writeFile(
    script,
    [
      '@echo off',
      'setlocal',
      ':wait',
      'timeout /t 1 /nobreak >nul',
      `tasklist /fi "IMAGENAME eq ${path.basename(currentExe)}" | find /i "${path.basename(currentExe)}" >nul && goto wait`,
      `move /y "${staged}" "${currentExe}" >nul`,
      `start "" "${currentExe}"`,
      `del "%~f0"`
    ].join('\r\n'),
    'utf8'
  )

  logger.info('update', 'staged update, relaunching', { staged })
  spawn('cmd.exe', ['/c', script], { detached: true, stdio: 'ignore', windowsHide: true }).unref()

  setTimeout(() => app.quit(), 500)
  return { ok: true, message: 'Update downloaded. The app will restart to apply it.' }
}
