/**
 * Ask-first updater.
 *
 * Checks a release feed, shows what changed, and applies only on the user's say-so: no silent
 * background updates, by decision. The llama.cpp backends ship inside the app bundle, so an app
 * update is a backend update.
 *
 * Installing hands over to the new version rather than swapping files behind the app's back: see
 * install.ts for the whole sequence and why. Shortcuts are looked after here too. Every pin made
 * from the window points at a stub that never moves, and pins made to an unpacked copy that has
 * since been removed get a forwarder in its place, so whatever the user clicks starts the newest
 * version.
 */

import { app, shell } from 'electron'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'
import crypto from 'node:crypto'
import { execFile, spawn } from 'node:child_process'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { logger } from '../log'
import {
  APP_EXE_NAME,
  FAILURE_MARKER,
  FORWARDER_MARKER,
  PENDING_FILE,
  PORTABLE_EXE_NAME,
  chooseUpdateTarget,
  compareVersions,
  isRuntimeExe,
  isStagedFor,
  looksLikeExe,
  misplacedLauncher,
  notReplacedText,
  notStartedText,
  parseDigest,
  parsePending,
  pendingVerdict,
  replaceWithRetry,
  stagedPath,
  type PendingUpdate
} from './install'

/** Where the portable launcher unpacks the app. */
function unpackCacheRoot(): string {
  return path.join(process.env.LOCALAPPDATA ?? path.join(os.homedir(), 'AppData', 'Local'), 'LLMManager')
}

/**
 * The entry that never moves: a small stub the launcher keeps at %LOCALAPPDATA%\LLMManager.
 *
 * It starts whichever portable exe last ran, wherever that is now, which then starts the newest
 * unpacked version. Pins and shortcuts aimed at it therefore survive updates and survive the user
 * moving the portable exe. Undefined until a launcher new enough to install it has run.
 */
export function stableEntry(): string | undefined {
  const stub = path.join(unpackCacheRoot(), APP_EXE_NAME)
  return fs.existsSync(stub) ? stub : undefined
}

/** Where the path of the portable exe is kept between runs. */
const ORIGIN_FILE = 'portable-origin.json'

/**
 * The portable exe this install is launched from: this run's, or the last one known.
 *
 * Written whenever a run is started from a portable exe outside the unpack cache, and read back by
 * runs that were not, which is what a pin to the unpacked app produces.
 */
export function portableOrigin(): string | undefined {
  const file = path.join(app.getPath('userData'), ORIGIN_FILE)
  const current = process.env.LLMM_PORTABLE_EXE
  const cache = unpackCacheRoot().toLowerCase() + path.sep
  // A download still waiting to take over is not where the app will live; the exe it replaces is.
  const waiting = current && /^\.update-/i.test(path.basename(current))
  if (current && !waiting && !current.toLowerCase().startsWith(cache) && fs.existsSync(current)) {
    try {
      fs.writeFileSync(file, JSON.stringify({ exe: current }))
    } catch {
      /* remembered next time */
    }
    return current
  }
  try {
    const saved = (JSON.parse(fs.readFileSync(file, 'utf8')) as { exe?: unknown }).exe
    return typeof saved === 'string' && fs.existsSync(saved) ? saved : undefined
  } catch {
    return undefined
  }
}

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
  /** The asset's SHA-256 as the release reports it; the download must match it to be installed. */
  sha256?: string | null
  publishedAt: string | null
  error?: string
  /**
   * Set when the last install did not finish.
   *
   * Reported separately from `error`, which describes this check. A previous failure is worth
   * saying even when the check itself went fine and an update is on offer, in fact especially
   * then, because the offer will look identical to the one that already failed once.
   */
  previousFailure?: string
}

function userFile(name: string): string {
  return path.join(app.getPath('userData'), name)
}

/**
 * Read the note a failed install leaves behind, if there is one.
 *
 * Deliberately does not clear it: `checkForUpdate` runs twice per apply, once for the user and
 * once to re-validate the URL, so consuming it here would let the internal call swallow the
 * notice before it was ever shown. It is cleared when a fresh attempt starts instead.
 */
function readFailureMarker(): string | null {
  try {
    const text = fs.readFileSync(userFile(FAILURE_MARKER), 'utf8').trim()
    if (!text) return null
    // Also in the log, so a diagnostics bundle shows it whether or not anyone opened Settings.
    logger.warn('update', 'the last update did not finish', { note: text.slice(0, 300) })
    return text.slice(0, 500)
  } catch {
    return null
  }
}

async function writeFailure(text: string): Promise<void> {
  await fsp.writeFile(userFile(FAILURE_MARKER), text, 'utf8').catch(() => undefined)
}

function readPending(): PendingUpdate | null {
  try {
    return parsePending(fs.readFileSync(userFile(PENDING_FILE), 'utf8'))
  } catch {
    return null
  }
}

async function clearPending(): Promise<void> {
  await fsp.rm(userFile(PENDING_FILE), { force: true }).catch(() => undefined)
}

export async function checkForUpdate(): Promise<UpdateInfo> {
  const currentVersion = app.getVersion()
  // Carried onto every outcome below: an install that failed last time is still true whether or
  // not the feed answers this time.
  const previousFailure = readFailureMarker() ?? undefined
  const none = { available: false, currentVersion, latestVersion: null, notes: null, downloadUrl: null, bytes: 0, publishedAt: null, previousFailure }
  try {
    const res = await fetch(UPDATE_FEED, {
      headers: { Accept: 'application/vnd.github+json', 'User-Agent': 'llm-manager' },
      signal: AbortSignal.timeout(15000)
    })
    if (!res.ok) return { ...none, error: `Update feed returned HTTP ${res.status}` }

    const release = (await res.json()) as {
      tag_name: string
      body?: string
      published_at?: string
      assets: { name: string; browser_download_url: string; size: number; digest?: string | null }[]
    }

    // The portable exe by name first: a release could carry other executables one day.
    const asset =
      release.assets.find((a) => a.name.toLowerCase() === PORTABLE_EXE_NAME.toLowerCase()) ??
      release.assets.find((a) => /\.exe$/i.test(a.name))
    const latestVersion = release.tag_name.replace(/^v/, '')

    return {
      available: compareVersions(latestVersion, currentVersion) > 0 && !!asset,
      currentVersion,
      latestVersion,
      notes: release.body ?? null,
      downloadUrl: asset?.browser_download_url ?? null,
      bytes: asset?.size ?? 0,
      sha256: parseDigest(asset?.digest),
      publishedAt: release.published_at ?? null,
      previousFailure
    }
  } catch (err) {
    return { ...none, error: err instanceof Error ? err.message : String(err) }
  }
}

/** Free bytes on the drive holding `dir`, or null when the system will not say. */
async function freeBytes(dir: string): Promise<number | null> {
  try {
    const s = await fsp.statfs(dir)
    return Number(s.bavail) * Number(s.bsize)
  } catch {
    return null
  }
}

const GB = 1024 ** 3

function gb(n: number): string {
  return `${(n / GB).toFixed(1)} GB`
}

/**
 * Download the new version beside the exe the user launches, check it, and hand over to it.
 *
 * The exe itself is left alone: the new version replaces it only after its window has loaded
 * (completeUpdate). Every failure up to the hand-off removes the download and leaves this version
 * running, exactly as it was.
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
   * What this function does with its argument is download it and run it, so whoever supplies the
   * argument decides what code runs as the user, permanently. The caller is the renderer, and the
   * same handler map serves the remote web UI, so "the caller is us" was never a safe assumption.
   * The only URL that may be applied is the one the release feed is offering right now.
   */
  const offered = await checkForUpdate()
  if (!offered.downloadUrl || url !== offered.downloadUrl || !offered.latestVersion) {
    return {
      ok: false,
      message: offered.error
        ? `Could not confirm the update against the release feed: ${offered.error}`
        : 'That download does not match the published release. Nothing was changed.'
    }
  }
  const version = offered.latestVersion

  /*
   * A real new attempt supersedes whatever the last one reported.
   *
   * Cleared here rather than when the notice is displayed, so the message survives until
   * something is actually done about it: closing Settings should not make it go away.
   */
  await fsp.rm(userFile(FAILURE_MARKER), { force: true }).catch(() => undefined)

  // Decided before downloading, so a copy that cannot be updated in place says so straight away.
  const chosen = chooseUpdateTarget({
    portableExe: process.env.LLMM_PORTABLE_EXE,
    portableDir: process.env.LLMM_PORTABLE_DIR,
    rememberedExe: portableOrigin(),
    appExe: app.getPath('exe'),
    cacheRoot: unpackCacheRoot(),
    exists: (file) => fs.existsSync(file)
  })
  if ('error' in chosen) return { ok: false, message: chosen.error }
  const target = chosen.target

  /*
   * Room for the download beside the exe, and for the new version to unpack.
   *
   * Checked up front so a full drive is a sentence now rather than a failed write most of a
   * gigabyte in. The unpack needs roughly three times the download (the archives, then what they
   * hold), and the launcher checks that again itself before it starts.
   */
  const need = offered.bytes > 0 ? offered.bytes : 1.2 * GB
  const freeHere = await freeBytes(path.dirname(target))
  if (freeHere !== null && freeHere < need + 256 * 1024 ** 2) {
    return { ok: false, message: `The update needs ${gb(need + 256 * 1024 ** 2)} free on the drive holding ${path.dirname(target)}, and it has ${gb(freeHere)}. Nothing was changed.` }
  }
  const freeCache = await freeBytes(path.dirname(unpackCacheRoot()))
  if (freeCache !== null && freeCache < need * 3) {
    return { ok: false, message: `The new version needs about ${gb(need * 3)} free on the drive holding ${unpackCacheRoot()} to unpack, and it has ${gb(freeCache)}. Nothing was changed.` }
  }

  const staged = stagedPath(target, version, Date.now())
  const partial = `${staged}.part`
  const discard = async (): Promise<void> => {
    await fsp.rm(partial, { force: true }).catch(() => undefined)
    await fsp.rm(staged, { force: true }).catch(() => undefined)
  }

  try {
    const res = await fetch(url, { redirect: 'follow' })
    if (!res.ok || !res.body) throw new Error(`Download failed: HTTP ${res.status}`)

    const total = offered.bytes || Number(res.headers.get('content-length') ?? 0)
    let done = 0
    const hash = crypto.createHash('sha256')
    let head = Buffer.alloc(0)
    const body = Readable.fromWeb(res.body as Parameters<typeof Readable.fromWeb>[0])
    body.on('data', (chunk: Buffer) => {
      done += chunk.length
      hash.update(chunk)
      if (head.length < 2) head = Buffer.concat([head, chunk.subarray(0, 2 - head.length)])
      onProgress({ done, total })
    })
    await pipeline(body, fs.createWriteStream(partial))

    // Every check passes before the file even gets the name the hand-off uses.
    if (total > 0 && done !== total) throw new Error(`The download was cut short (${done} of ${total} bytes).`)
    if (!looksLikeExe(head)) throw new Error('The download is not a Windows program.')
    const digest = hash.digest('hex')
    if (offered.sha256 && digest !== offered.sha256) {
      throw new Error('The download does not match the checksum the release publishes, so it was not installed.')
    }
    await fsp.rename(partial, staged)
    const written = await fileVersion(staged)
    if (written && compareVersions(written, version) !== 0) {
      throw new Error(`The download says it is version ${written}, not ${version}.`)
    }
  } catch (err) {
    await discard()
    const reason = err instanceof Error ? err.message : String(err)
    logger.warn('update', 'download rejected', { version, reason })
    return { ok: false, message: `${reason.replace(/\.?$/, '.')} Nothing was changed.` }
  }

  /*
   * Hand over to the new version.
   *
   * Recorded first, so that if the new version never comes up, this one can say so the next time
   * it starts (settleUnfinishedUpdate). The new version waits for this process to be gone before
   * it opens anything (waitForPreviousVersion), and replaces the exe only once its window has
   * loaded (completeUpdate).
   */
  const pending: PendingUpdate = { version, staged, target, from: app.getVersion(), at: Date.now() }
  await fsp.writeFile(userFile(PENDING_FILE), JSON.stringify(pending), 'utf8')

  const started = await new Promise<Error | null>((resolve) => {
    try {
      const child = spawn(staged, [], {
        detached: true,
        stdio: 'ignore',
        cwd: path.dirname(target),
        env: {
          ...process.env,
          LLMM_UPDATE_FROM_PID: String(process.pid),
          LLMM_UPDATE_TARGET: target,
          LLMM_UPDATE_VERSION: version
        }
      })
      child.once('spawn', () => {
        child.unref()
        resolve(null)
      })
      child.once('error', (err) => resolve(err))
    } catch (err) {
      resolve(err instanceof Error ? err : new Error(String(err)))
    }
  })
  if (started) {
    await discard()
    await clearPending()
    logger.warn('update', 'could not start the new version', { error: started.message })
    return { ok: false, message: `The new version could not be started (${started.message}). Nothing was changed.` }
  }

  logger.info('update', 'handed over to the new version', { version, staged, target })
  setTimeout(() => app.quit(), 300)
  return { ok: true, message: `Version ${version} is starting. This window will close.` }
}

/**
 * Before opening anything, let the version that started this one finish closing.
 *
 * Both would otherwise hold the database, the API port and llama-server at once. Bounded, so a
 * previous version that hangs on its way out costs a minute rather than this start.
 */
export async function waitForPreviousVersion(): Promise<void> {
  const pid = Number(process.env.LLMM_UPDATE_FROM_PID)
  delete process.env.LLMM_UPDATE_FROM_PID
  if (!Number.isInteger(pid) || pid <= 0 || pid === process.pid) return
  const deadline = Date.now() + 60_000
  while (Date.now() < deadline) {
    try {
      process.kill(pid, 0)
    } catch {
      return
    }
    await new Promise((resolve) => setTimeout(resolve, 200))
  }
  logger.warn('update', 'the previous version was still running after a minute; starting anyway', { pid })
}

/**
 * The new version takes the place of the old exe: called once its window has loaded.
 *
 * Only now is the exe the user launches replaced, by a rename within one folder, which Windows
 * does in one operation. If it cannot be done (something holding the file for a whole minute),
 * this version keeps running and the old exe stays usable; the note says so.
 */
export async function completeUpdate(): Promise<void> {
  const target = process.env.LLMM_UPDATE_TARGET
  const version = process.env.LLMM_UPDATE_VERSION ?? app.getVersion()
  const staged = process.env.LLMM_PORTABLE_EXE
  delete process.env.LLMM_UPDATE_TARGET
  delete process.env.LLMM_UPDATE_VERSION
  if (!app.isPackaged || !target || !staged) return
  if (!isStagedFor(staged, target)) {
    logger.warn('update', 'asked to install from a file that is not a download beside the exe; left alone', { staged, target })
    return
  }

  const replaced = await replaceWithRetry(staged, target, { rename: (from, to) => fsp.rename(from, to) })
  if (!replaced) {
    logger.warn('update', 'the new version could not replace the old exe', { staged, target })
    await writeFailure(notReplacedText(version, target))
    return
  }

  // From here on this run belongs to the real exe: its updates replace it, and pins lead to it.
  process.env.LLMM_PORTABLE_EXE = target
  portableOrigin()
  await clearPending()
  logger.info('update', 'installed', { version, target })
}

/**
 * What became of an install recorded by an earlier run, and old downloads nobody is waiting on.
 *
 * Run by every version at startup. The version that started an install sees, the next time it
 * runs, whether the new one ever came up, and says so if it did not, rather than offering the
 * same update again as if nothing had happened.
 */
export async function settleUnfinishedUpdate(): Promise<void> {
  if (!app.isPackaged) return
  const pending = readPending()
  if (pending) {
    const verdict = pendingVerdict(pending, app.getVersion(), Date.now())
    if (verdict === 'failed') {
      logger.warn('update', 'an update was downloaded but never started; this version was kept', { version: pending.version })
      if (isStagedFor(pending.staged, pending.target)) await fsp.rm(pending.staged, { force: true }).catch(() => undefined)
      if (!readFailureMarker()) await writeFailure(notStartedText(pending.version))
      await clearPending()
    } else if (verdict === 'arrived' && !process.env.LLMM_UPDATE_TARGET) {
      // A later run of the new version; completeUpdate has already done its part or reported why not.
      await clearPending()
    }
  }

  // Downloads left by an interrupted attempt, or by the batch-file updater this replaced.
  const origin = portableOrigin()
  if (!origin) return
  const keep = pending && pendingVerdict(pending, app.getVersion(), Date.now()) === 'starting' ? pending.staged.toLowerCase() : ''
  try {
    for (const name of await fsp.readdir(path.dirname(origin))) {
      if (!/^\.update-.+\.exe(\.part)?$/i.test(name)) continue
      const file = path.join(path.dirname(origin), name)
      if (file.toLowerCase() === keep || file.toLowerCase() === (process.env.LLMM_PORTABLE_EXE ?? '').toLowerCase()) continue
      const age = Date.now() - (await fsp.stat(file)).mtimeMs
      if (age > 60 * 60_000) await fsp.rm(file, { force: true }).catch(() => undefined)
    }
  } catch {
    /* nothing to tidy */
  }
}

/**
 * Keep every shortcut the user made leading to the newest version.
 *
 * Two kinds go stale. Pins made to an unpacked copy, before the window told the shell to relaunch
 * the launcher instead, point into a runtime folder a later version removed; those get a
 * forwarder put back where the app was, without touching the shortcut. Shortcuts to a portable
 * exe that has since been moved or deleted point at nothing; those are aimed at the stub, which
 * follows the launcher wherever it last ran. A shortcut that still works is never changed.
 */
export async function repairShortcuts(): Promise<void> {
  if (!app.isPackaged || process.platform !== 'win32') return
  const stub = stableEntry()
  if (!stub) return
  const cache = unpackCacheRoot()
  const roaming = app.getPath('appData')
  const places = [
    path.join(roaming, 'Microsoft', 'Internet Explorer', 'Quick Launch', 'User Pinned', 'TaskBar'),
    path.join(roaming, 'Microsoft', 'Internet Explorer', 'Quick Launch', 'User Pinned', 'StartMenu'),
    path.join(roaming, 'Microsoft', 'Internet Explorer', 'Quick Launch'),
    path.join(roaming, 'Microsoft', 'Windows', 'Start Menu', 'Programs'),
    app.getPath('desktop')
  ]

  const links: string[] = []
  const collect = async (dir: string, depth: number): Promise<void> => {
    let entries: fs.Dirent[]
    try {
      entries = await fsp.readdir(dir, { withFileTypes: true })
    } catch {
      return
    }
    for (const e of entries) {
      const full = path.join(dir, e.name)
      if (e.isFile() && e.name.toLowerCase().endsWith('.lnk')) links.push(full)
      else if (e.isDirectory() && depth > 0) await collect(full, depth - 1)
    }
  }
  for (const place of places) await collect(place, place.endsWith('Programs') ? 3 : 0)

  let forwarders = 0
  let retargeted = 0
  for (const link of new Set(links)) {
    let target: string
    let args = ''
    try {
      const details = shell.readShortcutLink(link)
      target = details.target
      args = details.args ?? ''
    } catch {
      continue
    }
    if (!target || fs.existsSync(target)) continue
    try {
      if (isRuntimeExe(target, cache)) {
        await fsp.mkdir(path.dirname(target), { recursive: true })
        await fsp.copyFile(stub, target)
        await fsp.writeFile(path.join(path.dirname(target), FORWARDER_MARKER), '')
        forwarders++
      } else if (path.basename(target).toLowerCase() === PORTABLE_EXE_NAME.toLowerCase()) {
        if (shell.writeShortcutLink(link, 'update', { target: stub, args, cwd: path.dirname(stub), icon: stub, iconIndex: 0 })) retargeted++
      }
    } catch (err) {
      logger.warn('update', 'could not repair a shortcut', { link, error: err instanceof Error ? err.message : String(err) })
    }
  }
  if (forwarders || retargeted) logger.info('update', 'repaired shortcuts to an old copy of the app', { forwarders, retargeted })
}

/** A file's product version, read through PowerShell; null when it cannot be read. */
function fileVersion(file: string): Promise<string | null> {
  return new Promise((resolve) => {
    execFile(
      path.join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe'),
      // The path goes in through the environment, so nothing about it can be read as a command.
      ['-NoProfile', '-NonInteractive', '-Command', '(Get-Item -LiteralPath $env:LLMM_VERSION_OF).VersionInfo.ProductVersion'],
      { windowsHide: true, timeout: 15000, env: { ...process.env, LLMM_VERSION_OF: file } },
      (err, stdout) => resolve(err ? null : String(stdout).trim() || null)
    )
  })
}

/** Delete the launcher an older updater left in the unpack cache, and the folder it sat in. */
async function removeStrayLauncher(file: string): Promise<void> {
  // The launcher can hold its own file for a moment after starting the app, so it is given a few tries.
  for (let attempt = 0; attempt < 10; attempt++) {
    try {
      await fsp.rm(file, { force: true })
      break
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 1000))
    }
  }
  const dir = path.dirname(file)
  // Never the folder this app is running from, nor one holding a model library: the same guard the launcher uses.
  if (path.relative(dir, path.dirname(app.getPath('exe'))) === '') return
  if (!/^runtime-/i.test(path.basename(dir)) || fs.existsSync(path.join(dir, 'LLMManagerModels'))) return
  await fsp.rm(dir, { recursive: true, force: true }).catch(() => undefined)
}

/**
 * Finish an update that an older version installed in the wrong place.
 *
 * Updaters before 1.6.6 replaced the unpacked copy of the app instead of the portable exe, so the
 * new version ran once and the next launch brought the old one back. When this run was started
 * that way (see misplacedLauncher), the launcher it came from is copied over the user's portable
 * exe, only if that exe is older than this version, and the stray copy is removed from the cache.
 */
export async function repairMisplacedUpdate(): Promise<void> {
  if (!app.isPackaged) return
  const misplaced = misplacedLauncher({
    portableExe: process.env.LLMM_PORTABLE_EXE,
    portableDir: process.env.LLMM_PORTABLE_DIR,
    cacheRoot: unpackCacheRoot(),
    exists: (file) => fs.existsSync(file)
  })
  if (!misplaced) return

  const { from, to } = misplaced
  try {
    const existing = await fileVersion(to)
    if (existing && compareVersions(existing, app.getVersion()) >= 0) {
      logger.info('update', 'a stray launcher was left in the unpack cache, but the portable exe is already current', { from, to, existing })
    } else {
      const temp = path.join(path.dirname(to), `.update-repair-${Date.now()}.exe`)
      await fsp.copyFile(from, temp)
      try {
        await fsp.rename(temp, to)
      } catch (err) {
        await fsp.rm(temp, { force: true }).catch(() => undefined)
        throw err
      }
      logger.info('update', 'finished an update an older version had installed in the wrong place', { from, to, replaced: existing })
    }
    // This run's own updates now go to the real exe.
    process.env.LLMM_PORTABLE_EXE = to
    // The stray launcher is most of a gigabyte, and nothing will start it again.
    await removeStrayLauncher(from)
  } catch (err) {
    logger.warn('update', 'could not finish an update an older version installed in the wrong place', {
      from,
      to,
      error: err instanceof Error ? err.message : String(err)
    })
  }
}
