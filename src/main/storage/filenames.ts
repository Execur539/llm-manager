/**
 * Turning user-supplied text into filenames Windows will accept.
 *
 * Kept separate from the database layer so it can be tested without a live SQLite handle,
 * and so the same rules apply anywhere the app writes a file named after something the user
 * typed — chat exports today, diagnostics bundles and transcripts later.
 */

import path from 'node:path'

/** Windows refuses these as filenames regardless of extension. */
const RESERVED_NAMES = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i

/** Characters Windows forbids outright, plus the C0 control range. */
// eslint-disable-next-line no-control-regex
const FORBIDDEN = /[<>:"/\\|?*\u0000-\u001f]+/g

/**
 * Build a safe filename stem from a title.
 *
 * A conversation title is free text. It can be entirely emoji, entirely punctuation, a reserved
 * device name, or end in a dot or space — which Windows silently strips, producing a file whose
 * name no longer matches what was asked for. Each of those cases previously produced either a
 * broken write or a bare ".md" with no stem at all.
 */
export function exportFilename(title: string, fallback: string): string {
  let safe = title
    .replace(FORBIDDEN, '-')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/[. ]+$/, '') // a trailing dot or space is dropped by the filesystem
    .slice(0, 60)
    .replace(/[. ]+$/, '') // slicing can expose a new trailing dot

  // Nothing usable survived — a title of "???" or "..." leaves no stem behind.
  if (!safe || /^[-.]+$/.test(safe)) safe = fallback
  if (RESERVED_NAMES.test(safe)) safe = `${safe}-file`
  return safe
}

/**
 * First free path in `dir` for the given stem and extension.
 *
 * Conversations are not uniquely named — several can be "New conversation" — and re-exporting
 * the same one is normal. Silently clobbering a previous export loses data the user believed
 * they had saved, so collisions take a numeric suffix instead.
 */
export function uniquePath(dir: string, stem: string, ext: string, exists: (p: string) => boolean): string {
  let file = path.join(dir, `${stem}.${ext}`)
  for (let i = 2; exists(file) && i < 1000; i++) {
    file = path.join(dir, `${stem} (${i}).${ext}`)
  }
  return file
}
