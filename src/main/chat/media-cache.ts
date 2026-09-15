/**
 * The images and audio a user message was sent with, kept so a rebuilt history can send them again.
 *
 * A message is stored as text. Its attachments reach the model as separate parts — base64 payloads
 * megabytes wide — which never go into the transcript. Anything that rebuilt a conversation from
 * storage therefore rebuilt it without them: every chat message after the one with the image, and
 * any agent turn resumed with Continue or rewound by an edit, went out with the earlier images gone,
 * and the model answered as though it had never seen them, or invented what they showed.
 *
 * Parts are kept here, by message, from the moment a turn is built. After a restart, or once the
 * oldest have been let go to stay within the memory cap, images and audio are rebuilt from their
 * attachment files. Video is not: sampling one again is minutes of work, so a video from before a
 * restart stays named in the text but is not sent again.
 */

import fs from 'node:fs'
import type { ModelCapabilities } from '@shared/types'
import type { ContentPart } from '../runtime/llama'
import { attachmentsFor } from './repo'

/** Room for a long session of screenshots, and a bound on what a very long one can hold onto. */
const MAX_CHARS = 512 * 1024 * 1024

const parts = new Map<string, ContentPart[]>()
let totalChars = 0

function sizeOf(list: ContentPart[]): number {
  return list.reduce(
    (n, p) => n + (p.image_url?.url.length ?? 0) + (p.input_audio?.data.length ?? 0) + (p.input_video?.data?.length ?? 0),
    0
  )
}

/** Keep the media parts a message was sent with. Its text is not kept here; the transcript has that. */
export function rememberMedia(messageId: string, sent: ContentPart[]): void {
  const media = sent.filter((p) => p.type !== 'text')
  const previous = parts.get(messageId)
  if (previous) {
    totalChars -= sizeOf(previous)
    parts.delete(messageId)
  }
  parts.set(messageId, media)
  totalChars += sizeOf(media)
  // Oldest first, which is the order a Map iterates in; a let-go message is rebuilt from disk if needed.
  for (const [id, list] of parts) {
    if (totalChars <= MAX_CHARS || id === messageId) break
    totalChars -= sizeOf(list)
    parts.delete(id)
  }
}

/**
 * The media a message was sent with, as far as the loaded model can take it.
 *
 * Filtered on the way out rather than when stored: the model can change between turns, and a server
 * refuses a whole request that carries an image it has no projector for.
 */
export function mediaFor(messageId: string, caps: ModelCapabilities | undefined): ContentPart[] | undefined {
  const list = parts.get(messageId)
  if (!list?.length || !caps) return undefined
  const usable = list.filter((p) => {
    if (p.type === 'image_url') return caps.vision
    if (p.type === 'input_audio') return caps.audio
    if (p.type === 'input_video') return caps.vision && caps.videoPossible
    return true
  })
  return usable.length ? usable : undefined
}

/**
 * Make sure the media for these messages is at hand, rebuilding what is missing from disk.
 *
 * `rebuild` turns attachment files into parts the way sending them did, or returns null when the
 * loaded model could not take them — in which case nothing is remembered, and the message is looked
 * at again once a model that can is loaded. A message with nothing to rebuild is remembered as such,
 * so it is not looked up again on every turn.
 */
export async function restoreMedia(
  messageIds: string[],
  rebuild: (files: string[]) => Promise<ContentPart[] | null>
): Promise<void> {
  for (const id of messageIds) {
    if (parts.has(id)) continue
    const files = attachmentsFor(id)
      .filter((a) => (a.kind === 'image' || a.kind === 'audio') && fs.existsSync(a.path))
      .map((a) => a.path)
    if (!files.length) {
      rememberMedia(id, [])
      continue
    }
    const rebuilt = await rebuild(files).catch(() => null)
    if (rebuilt) rememberMedia(id, rebuilt)
  }
}
