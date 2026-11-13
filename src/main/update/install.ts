/**
 * The parts of installing an update that need no Electron, so a test can run them on real files.
 *
 * How an update is installed:
 *
 * 1. The running version downloads the new portable exe into the folder of the one the user
 *    launches, under a hidden name, and checks it against the release: its size, its SHA-256 and
 *    the version written into it.
 * 2. It starts that download and quits. The download is a launcher like any other, so it unpacks
 *    itself and starts the new version.
 * 3. The new version, once its window has loaded, renames its own launcher over the old one.
 *
 * Nothing the user launches is replaced until the new version has shown that it runs. If it never
 * starts (a download the virus scanner quarantines, a drive too full to unpack it), the old exe is
 * still where it was, every shortcut still starts it, and the next check says what happened.
 *
 * This replaced a batch file that swapped the exe once the app had quit and then started it. The
 * swap came before anyone knew whether the new version worked, and cmd read any percent sign in
 * the folder's path as a variable.
 */

import path from 'node:path'

/** Where a failed install is recorded, relative to the app's userData directory. */
export const FAILURE_MARKER = 'update-failed.txt'

/** Where an install in progress is recorded, relative to the app's userData directory. */
export const PENDING_FILE = 'update-pending.json'

/** The portable exe's name, stable across versions since 1.0, so shortcuts to it keep working. */
export const PORTABLE_EXE_NAME = 'LLM-Manager-portable.exe'

/** The unpacked app's exe, and the stub that stands in for it once its folder is cleaned up. */
export const APP_EXE_NAME = 'LLM Manager.exe'

/** Marks a runtime folder that holds only a forwarder, so nothing mistakes it for an app. */
export const FORWARDER_MARKER = '.forwarder'

/**
 * How long a new version gets to come up before the old one calls the install failed.
 *
 * Its first start unpacks about a gigabyte, which takes a minute or two on a slow disk; ten
 * minutes is long past that and still short enough to report on the same sitting.
 */
export const START_GRACE_MS = 10 * 60_000

export function compareVersions(a: string, b: string): number {
  const pa = a.replace(/^v/, '').split(/[.-]/).map((x) => Number(x) || 0)
  const pb = b.replace(/^v/, '').split(/[.-]/).map((x) => Number(x) || 0)
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const x = pa[i] ?? 0
    const y = pb[i] ?? 0
    if (x !== y) return x - y
  }
  return 0
}

function isInside(root: string, target: string): boolean {
  const rel = path.relative(root, target)
  return !!rel && !rel.startsWith('..') && !path.isAbsolute(rel)
}

function samePath(a: string, b: string): boolean {
  return path.resolve(a).toLowerCase() === path.resolve(b).toLowerCase()
}

export interface UpdateTargetInput {
  /** LLMM_PORTABLE_EXE: the launcher that started this run, as it reports its own path. */
  portableExe?: string
  /** LLMM_PORTABLE_DIR: the folder the user's portable exe is in. */
  portableDir?: string
  /**
   * The portable exe an earlier run was started from, when this one was not started from any.
   *
   * A taskbar pin made while the app is running points at the unpacked copy, and a run started
   * from it is told nothing about the exe it came out of. That exe is still where it was.
   */
  rememberedExe?: string
  /** The running app's own executable. */
  appExe: string
  /** %LOCALAPPDATA%\LLMManager, where the launcher unpacks the app. */
  cacheRoot: string
  exists: (file: string) => boolean
}

/**
 * Which file an update replaces: the portable exe the user actually launches.
 *
 * Never the running app's own executable. In the portable build that is the copy unpacked under
 * LOCALAPPDATA, not the file anyone double-clicks, and replacing it is how updaters before 1.6.6
 * came to install a version that ran once. An install that was not started from a portable exe
 * cannot be updated in place at all, since what the release feed offers is a portable exe, so it
 * is told so rather than overwritten.
 */
export function chooseUpdateTarget(input: UpdateTargetInput): { target: string } | { error: string } {
  const { portableExe, portableDir, rememberedExe, appExe, cacheRoot, exists } = input
  const usable = (file: string | undefined): file is string =>
    !!file && !isInside(cacheRoot, file) && !/^\.update-/i.test(path.basename(file)) && exists(file)
  if (usable(portableExe)) return { target: portableExe }
  // Launchers before 1.6.6 passed their folder but not their own path; the name has not changed.
  if (portableDir && !isInside(cacheRoot, portableDir)) {
    const beside = path.join(portableDir, PORTABLE_EXE_NAME)
    if (exists(beside)) return { target: beside }
  }
  if (usable(rememberedExe)) return { target: rememberedExe }
  return {
    error: isInside(cacheRoot, appExe)
      ? `This copy was started from its unpacked files rather than from ${PORTABLE_EXE_NAME}, so there is no exe to update. Start the app from ${PORTABLE_EXE_NAME} and update again, or download the new version from the releases page.`
      : `This copy was not started from ${PORTABLE_EXE_NAME}, so there is nothing to update in place. Download the new version from the releases page.`
  }
}

/**
 * A launcher an older updater left in the unpack cache, and the user's exe it was meant to replace.
 *
 * Those updaters overwrote the unpacked app with the new launcher and started it from there. The
 * launcher keeps the real folder it was handed down, so the app it starts can finish the update by
 * copying the launcher over the exe the user launches. Null when this run did not come about that way.
 */
export function misplacedLauncher(input: Omit<UpdateTargetInput, 'appExe'>): { from: string; to: string } | null {
  const { portableExe, portableDir, cacheRoot, exists } = input
  if (!portableExe || !portableDir) return null
  if (!isInside(cacheRoot, portableExe) || isInside(cacheRoot, portableDir)) return null
  const to = path.join(portableDir, PORTABLE_EXE_NAME)
  return exists(portableExe) && exists(to) ? { from: portableExe, to } : null
}

/**
 * Where a download is kept until the new version takes over: beside the exe it will replace.
 *
 * The same folder makes the final step a rename within one drive, which Windows does in one
 * operation: there is never a moment with no exe at that path, or half of one. The leading dot and
 * the version keep it apart from the real exe and from another attempt's download.
 */
export function stagedPath(target: string, version: string, now: number): string {
  const safe = version.replace(/[^0-9A-Za-z.-]/g, '')
  return path.join(path.dirname(target), `.update-${safe}-${now}.exe`)
}

/** Whether a launcher is a download waiting to take the place of `target`, and nothing else. */
export function isStagedFor(staged: string, target: string): boolean {
  return (
    /^\.update-.+\.exe$/i.test(path.basename(staged)) &&
    samePath(path.dirname(staged), path.dirname(target)) &&
    !samePath(staged, target)
  )
}

/** The hex SHA-256 in a GitHub asset's `digest` field ("sha256:…"), or null if there is none. */
export function parseDigest(digest: unknown): string | null {
  if (typeof digest !== 'string') return null
  const m = /^sha256:([0-9a-f]{64})$/i.exec(digest.trim())
  return m ? m[1].toLowerCase() : null
}

/** The first bytes of every Windows executable. */
export function looksLikeExe(head: Uint8Array): boolean {
  return head.length >= 2 && head[0] === 0x4d && head[1] === 0x5a
}

export interface PendingUpdate {
  /** The version being installed. */
  version: string
  /** The download that will take the place of `target`. */
  staged: string
  /** The exe the user launches. */
  target: string
  /** The version that started the install. */
  from: string
  /** When the download was started, in ms since the epoch. */
  at: number
}

export function parsePending(text: string): PendingUpdate | null {
  try {
    const p = JSON.parse(text) as Partial<PendingUpdate>
    if (typeof p.version !== 'string' || typeof p.staged !== 'string' || typeof p.target !== 'string') return null
    if (typeof p.at !== 'number' || !Number.isFinite(p.at)) return null
    return { version: p.version, staged: p.staged, target: p.target, from: String(p.from ?? ''), at: p.at }
  } catch {
    return null
  }
}

/**
 * What a recorded install means to the version now running.
 *
 * - `arrived`: this is the new version or a later one; the record has done its job.
 * - `starting`: the new version may still be unpacking; leave it be.
 * - `failed`: the new version never came up, so the old one was kept.
 */
export function pendingVerdict(pending: PendingUpdate, currentVersion: string, now: number): 'arrived' | 'starting' | 'failed' {
  if (compareVersions(currentVersion, pending.version) >= 0) return 'arrived'
  return now - pending.at < START_GRACE_MS ? 'starting' : 'failed'
}

/**
 * Move the download over the exe, trying again while something holds the exe open.
 *
 * A virus scanner or Explorer's thumbnailer can hold a freshly written exe for a moment, and a
 * rename fails with a sharing violation for as long as it does. Retrying is all the waiting
 * needed: it fails while the file is held and succeeds the moment it is not.
 */
export async function replaceWithRetry(
  staged: string,
  target: string,
  opts: { tries?: number; delayMs?: number; rename: (from: string, to: string) => Promise<void> }
): Promise<boolean> {
  const tries = opts.tries ?? 60
  const delay = opts.delayMs ?? 1000
  for (let attempt = 0; attempt < tries; attempt++) {
    try {
      await opts.rename(staged, target)
      return true
    } catch {
      if (attempt < tries - 1) await new Promise((resolve) => setTimeout(resolve, delay))
    }
  }
  return false
}

/**
 * Whether a shortcut's target is an unpacked copy of the app: `<cache>\runtime-<version>\LLM Manager.exe`.
 *
 * Those folders are removed when a newer version unpacks, which used to leave every pin and
 * shortcut made to them pointing at nothing. One of these gets a forwarder put in its place.
 */
export function isRuntimeExe(target: string, cacheRoot: string): boolean {
  const dir = path.dirname(target)
  return (
    path.basename(target).toLowerCase() === APP_EXE_NAME.toLowerCase() &&
    /^runtime-/i.test(path.basename(dir)) &&
    samePath(path.dirname(dir), cacheRoot)
  )
}

/** Kept to one plain sentence or two: it is shown in Settings as it is. */
export function notStartedText(version: string): string {
  return `Version ${version} was downloaded but did not start, so it was not installed and this version was kept. Nothing else was changed. Try again, or download it from the releases page.`
}

export function notReplacedText(version: string, target: string): string {
  return `Version ${version} is running, but it could not take the place of ${target} because something was holding that file, so shortcuts still start the earlier version. Check for updates again to retry.`
}
