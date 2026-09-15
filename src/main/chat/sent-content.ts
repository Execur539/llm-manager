/**
 * What a message sent the model, in a form that can be stored and then sent again exactly.
 *
 * The transcript keeps what the user typed. What the model was sent is often more: the text of
 * attached documents, images, audio, a sampled video and its stills, an Ultra plan folded into the
 * prompt, a screenshot a tool returned. Rebuilding a history from the transcript alone dropped all
 * of it, so these are stored alongside — text as it was, and every media payload as a reference to a
 * file, so the database holds no base64 and the same bytes come back on the way out.
 *
 * Pure apart from the blob store it is handed, so the round trip can be tested without a disk.
 */

import type { ContentPart } from '../runtime/llama'

export type StoredPart =
  | { type: 'text'; text: string }
  /** `prefix` is everything before the comma of the data URL, so the URL is rebuilt exactly. */
  | { type: 'image'; file: string; prefix: string }
  | { type: 'audio'; file: string; format: string }
  | { type: 'video'; file: string }
  /** A part with nothing to move out, such as an image given as a web address, kept as it was. */
  | { type: 'inline'; part: ContentPart }

export type StoredContent = string | StoredPart[]

export interface BlobStore {
  /** Keep these bytes, returning the file they are in. */
  put(bytes: Buffer, ext: string): Promise<string>
  /** The bytes of a kept file, or null when it is gone. */
  get(file: string): Promise<Buffer | null>
}

const DATA_URL = /^(data:[^,]*;base64),(.*)$/s

export function parseDataUrl(url: string): { prefix: string; bytes: Buffer } | null {
  const match = DATA_URL.exec(url)
  return match ? { prefix: match[1], bytes: Buffer.from(match[2], 'base64') } : null
}

function extensionFor(prefix: string): string {
  const subtype = prefix.slice('data:'.length).split(';')[0].split('/')[1]?.toLowerCase() ?? ''
  if (subtype === 'jpeg') return 'jpg'
  if (subtype === 'svg+xml') return 'svg'
  return subtype.replace(/[^a-z0-9]/g, '') || 'bin'
}

/** Turn what was sent into its stored form, moving every media payload out into the blob store. */
export async function toStored(content: string | ContentPart[], blobs: BlobStore): Promise<StoredContent> {
  if (typeof content === 'string') return content
  const out: StoredPart[] = []
  for (const part of content) {
    if (part.type === 'text') {
      out.push({ type: 'text', text: part.text ?? '' })
    } else if (part.type === 'image_url' && part.image_url?.url) {
      const parsed = parseDataUrl(part.image_url.url)
      out.push(
        parsed
          ? { type: 'image', file: await blobs.put(parsed.bytes, extensionFor(parsed.prefix)), prefix: parsed.prefix }
          : { type: 'inline', part }
      )
    } else if (part.type === 'input_audio' && part.input_audio) {
      const { data, format } = part.input_audio
      out.push({ type: 'audio', file: await blobs.put(Buffer.from(data, 'base64'), format || 'bin'), format })
    } else if (part.type === 'input_video' && part.input_video?.data) {
      out.push({ type: 'video', file: await blobs.put(Buffer.from(part.input_video.data, 'base64'), 'mp4') })
    } else {
      out.push({ type: 'inline', part })
    }
  }
  return out
}

/**
 * Rebuild what was sent from its stored form.
 *
 * A part whose file has gone is left out and the rest still sent: a message missing one image is
 * better than a message missing everything it carried.
 */
export async function fromStored(stored: StoredContent, blobs: Pick<BlobStore, 'get'>): Promise<string | ContentPart[]> {
  if (typeof stored === 'string') return stored
  if (!Array.isArray(stored)) throw new Error('not stored content')
  const out: ContentPart[] = []
  for (const part of stored) {
    if (part.type === 'text') {
      out.push({ type: 'text', text: part.text })
    } else if (part.type === 'inline') {
      out.push(part.part)
    } else {
      const bytes = await blobs.get(part.file)
      if (!bytes) continue
      if (part.type === 'image') {
        out.push({ type: 'image_url', image_url: { url: `${part.prefix},${bytes.toString('base64')}` } })
      } else if (part.type === 'audio') {
        out.push({ type: 'input_audio', input_audio: { data: bytes.toString('base64'), format: part.format } })
      } else {
        out.push({ type: 'input_video', input_video: { data: bytes.toString('base64') } })
      }
    }
  }
  return out
}

/**
 * What of a message the loaded model can take.
 *
 * A server refuses a whole request that carries an image it has no projector for, so anything the
 * model cannot read is left out rather than sent.
 */
export function forCaps(
  content: ContentPart[],
  caps: { vision: boolean; audio: boolean; videoPossible: boolean } | undefined
): ContentPart[] {
  return content.filter((p) => {
    if (p.type === 'image_url') return !!caps?.vision
    if (p.type === 'input_audio') return !!caps?.audio
    if (p.type === 'input_video') return !!caps?.vision && !!caps?.videoPossible
    return true
  })
}

/** The text a message sent, attached documents included. */
export function textOf(content: string | ContentPart[]): string {
  if (typeof content === 'string') return content
  return content
    .filter((p) => p.type === 'text')
    .map((p) => p.text ?? '')
    .join('\n\n')
}
