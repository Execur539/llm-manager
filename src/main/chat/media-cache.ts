/**
 * What each message sent the model beyond the text the transcript stores, so a rebuilt history
 * sends the same.
 *
 * The transcript keeps what the user typed. The model was often sent more — attached documents'
 * text, images, audio, a sampled video, an Ultra plan folded into the prompt, a screenshot a tool
 * returned — and anything that rebuilt a conversation from storage rebuilt it without them: every
 * chat message after the one with the attachment, every agent turn, Continue, an edit, a restart.
 * The model answered as though it had never seen them, or invented what they showed.
 *
 * What was sent is recorded when it is sent: in memory for this run, and in the database with its
 * media as kept files, so a restart sends it again byte for byte. Messages from before this was
 * recorded are rebuilt from their kept attachment files once, by the caller's `reconstruct`.
 */

import type { AgentMessage, ModelCapabilities } from '@shared/types'
import type { ContentPart } from '../runtime/llama'
import { logger } from '../log'
import { attachmentRowsForMessage, loadSentContent, saveSentContent } from './repo'
import { sentBlobs } from './attachment-store'
import { forCaps, fromStored, textOf, toStored, type StoredContent } from './sent-content'

/** Room for a long session of screenshots, and a bound on what a very long one can hold onto. */
const MAX_CHARS = 512 * 1024 * 1024

/** Null records that a message sent nothing beyond its stored text, so it is not looked up again. */
type Sent = string | ContentPart[] | null

const cache = new Map<string, Sent>()
let totalChars = 0

function sizeOf(content: Sent): number {
  if (!content) return 0
  if (typeof content === 'string') return content.length
  return content.reduce(
    (n, p) =>
      n +
      (p.text?.length ?? 0) +
      (p.image_url?.url.length ?? 0) +
      (p.input_audio?.data.length ?? 0) +
      (p.input_video?.data?.length ?? 0),
    0
  )
}

function remember(messageId: string, content: Sent): void {
  const previous = cache.get(messageId)
  if (previous !== undefined) {
    totalChars -= sizeOf(previous)
    cache.delete(messageId)
  }
  cache.set(messageId, content)
  totalChars += sizeOf(content)
  // Oldest first, which is the order a Map iterates in; a message let go is read back from storage.
  for (const [id, entry] of cache) {
    if (totalChars <= MAX_CHARS || id === messageId) break
    totalChars -= sizeOf(entry)
    cache.delete(id)
  }
}

/**
 * What a message sent, as far as the loaded model can take it; undefined when it sent only its
 * stored text, or has not been restored.
 */
export function sentContentFor(messageId: string, caps: ModelCapabilities | undefined): string | ContentPart[] | undefined {
  const content = cache.get(messageId)
  if (content === undefined || content === null) return undefined
  return typeof content === 'string' ? content : forCaps(content, caps)
}

/** The text of what a message sent, attached documents included, when more than its stored text. */
export function sentTextFor(messageId: string): string | undefined {
  const content = cache.get(messageId)
  return content ? textOf(content) : undefined
}

/**
 * Record what a message sent: at once in memory, then in the database with its media as kept files.
 *
 * Bytes identical to one of the message's kept attachments point at that file rather than being
 * written a second time. A failure to store is logged, not thrown: the message has already gone.
 */
export async function persistSent(chatId: string, messageId: string, content: string | ContentPart[]): Promise<void> {
  remember(messageId, content)
  try {
    const known = attachmentRowsForMessage(messageId).flatMap((row) =>
      // A video's own file is the source clip, possibly gigabytes; only what was made from it was sent.
      row.kind === 'video' ? [...(row.meta.optimised ? [row.meta.optimised] : []), ...(row.meta.stills ?? [])] : [row.path]
    )
    const stored = await toStored(content, sentBlobs(chatId, known))
    saveSentContent(chatId, messageId, JSON.stringify(stored))
  } catch (err) {
    logger.warn('attachments', `could not record what message ${messageId} sent`, err)
  }
}

/**
 * Make sure what a conversation's messages sent is at hand for rebuilding its history.
 *
 * Read from the database where it was recorded. A user message from before recording began is
 * rebuilt by `reconstruct` from its kept files and recorded, which returns null when it carried
 * nothing beyond its text, and undefined when the loaded model cannot take what it carried — in
 * which case it is looked at again once one that can is loaded.
 */
export async function restoreSent(
  chatId: string,
  messages: AgentMessage[],
  reconstruct: (message: AgentMessage) => Promise<ContentPart[] | null | undefined>
): Promise<void> {
  const blobs = sentBlobs(chatId, [])
  for (const message of messages) {
    if (message.role !== 'user' && message.role !== 'tool') continue
    if (cache.has(message.id)) continue

    const raw = loadSentContent(message.id)
    if (raw !== null) {
      try {
        remember(message.id, await fromStored(JSON.parse(raw) as StoredContent, blobs))
        continue
      } catch (err) {
        logger.warn('attachments', `could not read what message ${message.id} sent`, err)
      }
    }

    // A tool result with no record returned no media.
    if (message.role === 'tool') {
      remember(message.id, null)
      continue
    }
    const rebuilt = await reconstruct(message).catch(() => undefined)
    if (rebuilt === undefined) continue
    if (rebuilt === null) remember(message.id, null)
    else await persistSent(chatId, message.id, rebuilt)
  }
}
