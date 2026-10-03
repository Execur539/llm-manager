/**
 * The folders the library scans.
 *
 * The app's own models folder is where downloads and imports land, and it was the only place the
 * library looked. Models kept elsewhere — a second drive set aside for AI files, say — could only
 * be brought in by copying them, which for a 100 GB model costs 100 GB on a drive that may not
 * have it. Extra folders are scanned in place instead: their models are listed and loaded from
 * where they are, and nothing is ever written into them.
 */

import path from 'node:path'

/** More than anyone keeps by hand, and a bound on how much a hand-edited settings file can make the scan walk. */
const MAX_EXTRA_DIRS = 32

const fold = (p: string): string => (process.platform === 'win32' ? p.toLowerCase() : p)

/** Whether `file` is `dir` itself or anywhere beneath it. */
export function isInside(file: string, dir: string): boolean {
  const rel = path.relative(fold(path.resolve(dir)), fold(path.resolve(file)))
  // `..\Models2` must not count as inside `Models`, and a path on another drive comes back absolute.
  return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel))
}

/**
 * The stored list of extra folders, made safe to walk.
 *
 * The settings file is plain JSON a person can edit, so anything can be in it: a single string
 * where a list belongs, numbers, a relative path that would resolve against whatever the working
 * directory happens to be. Only absolute paths survive, each once.
 */
export function sanitizeModelDirs(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  const out: string[] = []
  const seen = new Set<string>()
  for (const v of value) {
    if (typeof v !== 'string' || !v.trim() || !path.isAbsolute(v.trim())) continue
    const dir = path.resolve(v.trim())
    if (seen.has(fold(dir))) continue
    seen.add(fold(dir))
    out.push(dir)
    if (out.length >= MAX_EXTRA_DIRS) break
  }
  return out
}

/**
 * The app's models folder first, then each extra folder not already covered by one before it.
 *
 * A folder inside another would list its models twice. An extra folder that contains the models
 * folder is kept, since it holds models of its own; the scan drops the files it finds twice.
 */
export function libraryRoots(modelsDir: string, extra: readonly string[]): string[] {
  const roots = [path.resolve(modelsDir)]
  for (const dir of sanitizeModelDirs(extra)) {
    if (roots.some((r) => isInside(dir, r))) continue
    roots.push(dir)
  }
  return roots
}
