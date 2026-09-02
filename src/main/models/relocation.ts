/**
 * Models-folder relocation.
 *
 * The exe is portable. When it moves, the models folder is left behind — so on launch we
 * compare the breadcrumb in %APPDATA% against where the exe now sits and, if they disagree,
 * offer to bring the models across.
 *
 * Two paths matter:
 *   same volume  -> fs.rename is a directory-entry update: instant even for 100 GB.
 *   cross volume -> a real byte-for-byte copy, so it needs progress, cancellation, and
 *                   must never delete the source until the copy is verified.
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import {
  defaultModelsDir,
  exeDir,
  exeLocationUnknown,
  isInsideRuntimeCache,
  readBreadcrumb,
  writeBreadcrumb
} from '../storage/paths'
import { loadSettings } from '../storage/settings'
import { logger } from '../log'

export interface RelocationProposal {
  /** where the models are now */
  from: string
  /** where they would go (beside the current exe) */
  to: string
  /**
   * The folder the app itself is in.
   *
   * Carried separately because `to` is a models folder, not the application's location, and a
   * dialog that shows one while claiming the other is worse than showing neither.
   */
  appDir: string
  totalBytes: number
  fileCount: number
  /** true when the move is a same-volume rename and therefore effectively instant */
  sameVolume: boolean
}

export interface MoveProgress {
  copiedBytes: number
  totalBytes: number
  currentFile: string
  done: boolean
  cancelled: boolean
  error?: string
}

function volumeOf(p: string): string {
  return path.parse(path.resolve(p)).root.toLowerCase()
}

async function dirStats(dir: string): Promise<{ bytes: number; files: number }> {
  let bytes = 0
  let files = 0
  async function walk(d: string): Promise<void> {
    let entries: fs.Dirent[]
    try {
      entries = await fsp.readdir(d, { withFileTypes: true })
    } catch {
      return
    }
    for (const e of entries) {
      const full = path.join(d, e.name)
      if (e.isDirectory()) await walk(full)
      else if (e.isFile()) {
        try {
          bytes += (await fsp.stat(full)).size
          files++
        } catch {
          /* skip */
        }
      }
    }
  }
  await walk(dir)
  return { bytes, files }
}

/**
 * Decide whether a relocation should be offered.
 * Returns null when nothing needs to happen (first run, or models already beside the exe).
 */
export async function checkRelocation(): Promise<RelocationProposal | null> {
  /*
   * Nothing to propose if we do not reliably know where the app is.
   *
   * Running the unpacked copy directly — which is what a taskbar pin does, since Windows
   * resolves the pin to the executable it sees rather than to the launcher — leaves
   * `exeDir()` pointing at the extraction cache and `defaultModelsDir()` falling back to
   * Documents. Comparing a real library against that fallback concluded the models were
   * misplaced and offered to copy them, which in the case that found this was seventeen
   * gigabytes moved off a drive they were already correctly on.
   */
  if (exeLocationUnknown()) {
    logger.warn(
      'relocation',
      'running from the extraction cache with no portable directory — skipping the models-location check',
      { hint: 'launch the portable exe rather than the unpacked copy' }
    )
    return null
  }

  /*
   * An explicitly chosen models folder settles the question.
   *
   * Relocation asks "are the models beside the app", but that is only the right question when
   * nobody has answered it already. Someone who has set a folder in Settings — or answered
   * "Keep them there" once — has stated where their library belongs, and moving the exe does
   * not revoke that. This checked the computed default against the breadcrumb and never looked
   * at the setting, so it proposed moving libraries that were exactly where they had been put.
   */
  const configured = loadSettings().modelsDir
  if (configured && fs.existsSync(configured)) return null

  const target = defaultModelsDir()

  // Never propose moving a library into the portable extraction cache. That directory is
  // deleted on upgrade, so "bringing the models along" would quietly stage them for deletion.
  if (isInsideRuntimeCache(target)) return null

  const crumb = readBreadcrumb()

  if (!crumb) {
    // First run: adopt the default location silently.
    await fsp.mkdir(target, { recursive: true })
    writeBreadcrumb(target)
    return null
  }

  const from = crumb.modelsDir
  if (path.resolve(from).toLowerCase() === path.resolve(target).toLowerCase()) return null
  if (!fs.existsSync(from)) {
    // The old folder is gone — nothing to move; re-point at the new location.
    await fsp.mkdir(target, { recursive: true })
    writeBreadcrumb(target)
    return null
  }

  const { bytes, files } = await dirStats(from)
  if (files === 0) {
    writeBreadcrumb(target)
    return null
  }

  return {
    from,
    to: target,
    appDir: exeDir(),
    totalBytes: bytes,
    fileCount: files,
    sameVolume: volumeOf(from) === volumeOf(target)
  }
}

/** Keep using the models where they are; just remember that decision. */
export function keepInPlace(from: string): void {
  writeBreadcrumb(from)
}

/**
 * Perform the move. `onProgress` is called as bytes land; `shouldCancel` is polled between
 * files so a long cross-volume copy can be abandoned without leaving a half-state.
 */
export async function performMove(
  proposal: RelocationProposal,
  onProgress: (p: MoveProgress) => void,
  shouldCancel: () => boolean
): Promise<MoveProgress> {
  const { from, to, totalBytes } = proposal

  if (proposal.sameVolume) {
    try {
      await fsp.mkdir(path.dirname(to), { recursive: true })
      await fsp.rename(from, to)
      writeBreadcrumb(to)
      const done: MoveProgress = { copiedBytes: totalBytes, totalBytes, currentFile: '', done: true, cancelled: false }
      onProgress(done)
      return done
    } catch {
      // Rename can still fail across mount points that look like the same volume; fall through to copy.
    }
  }

  let copied = 0
  const copiedFiles: string[] = []

  async function copyDir(srcDir: string, dstDir: string): Promise<boolean> {
    await fsp.mkdir(dstDir, { recursive: true })
    const entries = await fsp.readdir(srcDir, { withFileTypes: true })
    for (const e of entries) {
      if (shouldCancel()) return false
      const src = path.join(srcDir, e.name)
      const dst = path.join(dstDir, e.name)
      if (e.isDirectory()) {
        if (!(await copyDir(src, dst))) return false
      } else if (e.isFile()) {
        onProgress({ copiedBytes: copied, totalBytes, currentFile: e.name, done: false, cancelled: false })
        await fsp.copyFile(src, dst)
        const st = await fsp.stat(dst)
        const srcSt = await fsp.stat(src)
        // Verify before the source is ever considered removable.
        if (st.size !== srcSt.size) {
          throw new Error(`Copy verification failed for ${e.name} (${st.size} != ${srcSt.size})`)
        }
        copied += st.size
        copiedFiles.push(dst)
      }
    }
    return true
  }

  try {
    const completed = await copyDir(from, to)
    if (!completed) {
      // Cancelled: remove what we copied so the destination isn't a partial library.
      for (const f of copiedFiles.reverse()) {
        try {
          await fsp.rm(f, { force: true })
        } catch {
          /* best effort */
        }
      }
      const cancelled: MoveProgress = { copiedBytes: copied, totalBytes, currentFile: '', done: false, cancelled: true }
      onProgress(cancelled)
      return cancelled
    }

    // Only now, with every file copied and size-verified, is it safe to drop the source.
    await fsp.rm(from, { recursive: true, force: true })
    writeBreadcrumb(to)
    const done: MoveProgress = { copiedBytes: copied, totalBytes, currentFile: '', done: true, cancelled: false }
    onProgress(done)
    return done
  } catch (err) {
    const failed: MoveProgress = {
      copiedBytes: copied,
      totalBytes,
      currentFile: '',
      done: false,
      cancelled: false,
      error: err instanceof Error ? err.message : String(err)
    }
    onProgress(failed)
    return failed
  }
}
