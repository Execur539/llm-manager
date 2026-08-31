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
import { buildSwapScript, FAILURE_MARKER } from './swap-script'

/** Where releases are published. Overridable so a fork can point elsewhere. */
const UPDATE_FEED =
  process.env.LLMM_UPDATE_FEED ?? 'https://api.github.com/repos/Execur539/llm-manager/releases/latest'

export interface UpdateInfo {
  available: boolean
  currentVersion: string
  latestVersion: string | null
  notes: string | null
  downloadUrl: string | null
  bytes: number
  publishedAt: string | null
  error?: string
  /**
   * Set when the last attempt downloaded successfully but could not swap the exe.
   *
   * Reported separately from `error`, which describes this check. A previous failure is worth
   * saying even when the check itself went fine and an update is on offer — in fact especially
   * then, because the offer will look identical to the one that already failed once.
   */
  previousFailure?: string
}

/**
 * Read the marker the swap helper leaves behind, if there is one.
 *
 * Deliberately does not clear it: `checkForUpdate` runs twice per apply — once for the user and
 * once to re-validate the URL — so consuming it here would let the internal call swallow the
 * notice before it was ever shown. It is cleared when a fresh attempt starts instead.
 */
function readFailureMarker(): string | null {
  try {
    const text = fs.readFileSync(path.join(app.getPath('userData'), FAILURE_MARKER), 'utf8').trim()
    if (!text) return null
    // Also in the log, so a diagnostics bundle shows it whether or not anyone opened Settings.
    logger.warn('update', 'a previous update downloaded but could not replace the exe')
    return text.slice(0, 500)
  } catch {
    return null
  }
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
  // Carried onto every outcome below: a swap that failed last time is still true whether or not
  // the feed answers this time.
  const previousFailure = readFailureMarker() ?? undefined
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
        previousFailure,
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
      publishedAt: release.published_at ?? null,
      previousFailure
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
      previousFailure,
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

  /*
   * The URL is checked against the feed rather than trusted.
   *
   * What this function does with its argument is download it, overwrite the running executable
   * with it, and launch it — so whoever supplies the argument decides what code runs as the user,
   * permanently. The caller is the renderer, and the same handler map serves the remote web UI,
   * so "the caller is us" was never a safe assumption. The only URL that may be applied is the
   * one the release feed is offering right now.
   */
  const offered = await checkForUpdate()
  if (!offered.downloadUrl || url !== offered.downloadUrl) {
    return {
      ok: false,
      message: offered.error
        ? `Could not confirm the update against the release feed: ${offered.error}`
        : 'That download does not match the published release. Nothing was changed.'
    }
  }

  /*
   * A real new attempt supersedes whatever the last one reported.
   *
   * Cleared here rather than when the notice is displayed, so the message survives until
   * something is actually done about it — closing Settings should not make it go away.
   */
  await fsp.rm(path.join(app.getPath('userData'), FAILURE_MARKER), { force: true }).catch(() => undefined)

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

  /*
   * Windows will not let a running exe overwrite itself, so a helper does it after exit.
   *
   * The helper retries the move rather than waiting for the process to disappear first. That is
   * the whole synchronisation: the move fails with a sharing violation while the file is locked
   * and succeeds the moment it is not, so there is nothing to poll and nothing to match on.
   *
   * The previous version polled `tasklist` for the image name, which had two problems. It could
   * not tell this copy from another portable copy elsewhere on disk with the same filename — and
   * every copy now has the same filename by design. And it slept with `timeout`, which refuses
   * to run at all when stdin is redirected ("Input redirection is not supported"), which is
   * exactly what `stdio: 'ignore'` below does — so the sleep returned instantly and the wait
   * loop was a busy spin. `ping` has no such objection, and both are called by absolute path so
   * a shadowing entry earlier in PATH cannot substitute a different program.
   */
  const marker = path.join(app.getPath('userData'), FAILURE_MARKER)
  const script = path.join(os.tmpdir(), `llmm-update-${Date.now()}.cmd`)
  await fsp.writeFile(script, buildSwapScript({ target: currentExe, staged, marker }), 'utf8')

  logger.info('update', 'staged update, relaunching', { staged })
  spawn('cmd.exe', ['/c', script], { detached: true, stdio: 'ignore', windowsHide: true }).unref()

  setTimeout(() => app.quit(), 500)
  return { ok: true, message: 'Update downloaded. The app will restart to apply it.' }
}
