/**
 * Letting the agent look at things.
 *
 * The tool set could produce images and never read one. `screenshot` saved a PNG and returned a
 * path; `browser_screenshot` did the same; the clipboard reader handled text only. Meanwhile the
 * same vision model reads images perfectly well in Chat, because that path builds content parts.
 * `screenshot`'s own description told the model to "use read_image", which did not exist.
 *
 * These return media parts alongside their text, and the loop feeds them to the model as an
 * image turn. Everything here degrades to a plain explanation when the loaded model has no
 * vision projector, rather than silently handing back something nothing will look at.
 */

import fsp from 'node:fs/promises'
import path from 'node:path'
import { clipboard, nativeImage } from 'electron'
import type { Tool, ToolOutput } from './base'
import { schema, str, int } from './base'
import type { ContentPart } from '../../runtime/llama'
import { extractText } from '../../rag'
import * as rag from '../../rag'

/**
 * Longest edge an image is scaled to before it is sent.
 *
 * A 4K screenshot is about 24 MB of base64 and thousands of image tokens, most of them spent on
 * pixels no vision encoder resolves anyway — projectors work from a fixed patch grid. Downscaling
 * costs nothing that matters and keeps a single screenshot from eating the context window.
 * Electron's own image decoder does it, so no extra binary is involved.
 */
const MAX_EDGE = 1568

const IMAGE_EXT = new Set(['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'])

/** Decode, downscale if oversized, and package as an image content part. */
function toPart(image: Electron.NativeImage): { part: ContentPart; note: string } {
  const { width, height } = image.getSize()
  const longest = Math.max(width, height)

  const sized = longest > MAX_EDGE ? image.resize(width >= height ? { width: MAX_EDGE } : { height: MAX_EDGE }) : image
  const out = sized.getSize()
  const note =
    longest > MAX_EDGE
      ? `${width}x${height}, scaled to ${out.width}x${out.height} for the model`
      : `${width}x${height}`

  return {
    part: { type: 'image_url', image_url: { url: `data:image/png;base64,${sized.toPNG().toString('base64')}` } },
    note
  }
}

const noVision =
  'The loaded model has no vision projector, so it cannot be shown an image. Load a multimodal ' +
  'model (one with an mmproj file) to use this, or work from the file path instead.'

const readImage: Tool = {
  name: 'read_image',
  description:
    'Look at an image file — a screenshot you just took, a page capture, a picture on disk. The ' +
    'image is shown to you directly; describe or reason about what you see. Requires a model ' +
    'with vision. Large images are scaled down automatically.',
  tier: 'read',
  parameters: schema({ path: str('Path to a PNG, JPG, GIF, WEBP or BMP file') }, ['path']),
  async run(args, ctx): Promise<ToolOutput> {
    const file = path.isAbsolute(String(args.path))
      ? path.normalize(String(args.path))
      : path.resolve(ctx.cwd, String(args.path))

    if (!IMAGE_EXT.has(path.extname(file).toLowerCase())) {
      return `${file} does not look like an image. Supported: ${[...IMAGE_EXT].join(', ')}.`
    }
    if (!ctx.vision) return noVision

    const stat = await fsp.stat(file)
    if (stat.size > 64 * 1024 * 1024) {
      return `${file} is ${(stat.size / 1e6).toFixed(0)} MB — too large to decode as an image.`
    }

    const image = nativeImage.createFromPath(file)
    if (image.isEmpty()) return `${file} could not be decoded as an image.`

    const { part, note } = toPart(image)
    return { text: `Showing ${path.basename(file)} (${note}).`, media: [part] }
  }
}

const readClipboardImage: Tool = {
  name: 'read_clipboard_image',
  description:
    'Look at an image on the clipboard — what a Print Screen or a snipping tool just copied. ' +
    'Use read_clipboard for text. Requires a model with vision.',
  tier: 'read',
  parameters: schema({}),
  async run(_args, ctx): Promise<ToolOutput> {
    if (!ctx.vision) return noVision

    const image = clipboard.readImage()
    if (image.isEmpty()) {
      return 'The clipboard holds no image. If you expected text, use read_clipboard instead.'
    }
    const { part, note } = toPart(image)
    return { text: `Showing the clipboard image (${note}).`, media: [part] }
  }
}

const readDocument: Tool = {
  name: 'read_document',
  description:
    'Extract the text of a PDF or other document format. Use this rather than read_file for ' +
    'anything that is not plain text — read_file returns unreadable bytes for a PDF.',
  tier: 'read',
  parameters: schema(
    { path: str('Path to the document'), max_chars: int('Truncate the extracted text (default 60000)') },
    ['path']
  ),
  async run(args, ctx) {
    const file = path.isAbsolute(String(args.path))
      ? path.normalize(String(args.path))
      : path.resolve(ctx.cwd, String(args.path))

    const text = await extractText(file)
    const limit = Math.max(1000, Number(args.max_chars ?? 60000))
    const trimmed = text.trim()
    if (!trimmed) return `${path.basename(file)} contained no extractable text.`

    return trimmed.length > limit
      ? `${trimmed.slice(0, limit)}\n\n[${trimmed.length - limit} more characters; call again with a higher max_chars or a narrower target]`
      : trimmed
  }
}

const searchDocuments: Tool = {
  name: 'search_documents',
  description:
    "Search the user's document collections by meaning, not keyword. These are files they have " +
    'deliberately embedded for retrieval. Use it when a question refers to their own documents, ' +
    'notes or reference material rather than to the filesystem or the web.',
  tier: 'read',
  parameters: schema(
    {
      query: str('What to look for, phrased as the question you want answered'),
      collection: str('Restrict to one collection by name; omit to search them all'),
      limit: int('How many passages to return (default 6)')
    },
    ['query']
  ),
  async run(args, ctx) {
    const collections = rag.listCollections()
    if (!collections.length) {
      return 'There are no document collections yet. The user creates them in the Documents tab.'
    }

    const wanted = args.collection ? String(args.collection).toLowerCase() : null
    const targets = wanted
      ? collections.filter((c) => c.name.toLowerCase() === wanted || c.id === args.collection)
      : collections

    if (wanted && !targets.length) {
      return `No collection named "${args.collection}". Available: ${collections.map((c) => c.name).join(', ')}.`
    }

    const limit = Math.max(1, Math.min(20, Number(args.limit ?? 6)))

    /*
     * Retrieval is scoped per collection, so searching "everywhere" means asking each in turn
     * and merging. `retrieve` deliberately returns nothing for an empty scope — an unscoped
     * search used to mean "scan the entire database", which is not a question anyone asked.
     */
    const hits = (
      await Promise.all(targets.map((c) => rag.retrieve(String(args.query), ctx.backend, { collectionId: c.id }, limit)))
    ).flat()

    if (!hits.length) return `Nothing in ${targets.map((c) => c.name).join(', ')} matched that.`

    return hits
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map((h) => `--- ${h.filename} (chunk ${h.ord}, score ${h.score.toFixed(3)}) ---\n${h.text}`)
      .join('\n\n')
  }
}

export const visionTools: Tool[] = [readImage, readClipboardImage, readDocument, searchDocuments]
