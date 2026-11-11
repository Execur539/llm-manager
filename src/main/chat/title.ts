/**
 * Sidebar titles from a model's answer.
 *
 * Pure, so the smoke suite can hold it to the answers models actually give: quoted, labelled,
 * bolded, prefaced with a heading, or echoing a prefilled empty thought block.
 */

export const TITLE_MAX = 60

/** The first line of a model's answer, without the quoting, markup and labels models add. */
export function cleanTitle(text: string): string {
  const line =
    text
      .replace(/<think>[\s\S]*?<\/think>/gi, '')
      .split('\n')
      .map((l) => l.trim())
      .find((l) => l.length > 0) ?? ''
  let title = line
    .replace(/^#+\s*/, '')
    .replace(/^(?:title|name)\s*:\s*/i, '')
    .replace(/[*_`]/g, '')
    .replace(/^["'\u201c\u2018]+|["'\u201d\u2019]+$/g, '')
    .replace(/[.:;,]+$/, '')
    .replace(/\s+/g, ' ')
    .trim()
  if (title.length > TITLE_MAX) {
    const cut = title.slice(0, TITLE_MAX)
    const space = cut.lastIndexOf(' ')
    title = (space > TITLE_MAX / 2 ? cut.slice(0, space) : cut).trim()
  }
  return title
}
