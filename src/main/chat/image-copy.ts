/**
 * How the app's kept copy of an attached image is made: as it came, or reduced.
 *
 * A vision projector does not read an image at any size it is given: llama.cpp resizes it to the
 * projector's own limit first, and pixels past that are uploaded and held in memory on every turn
 * for nothing. Most of llama.cpp's projectors stop at 4,096 image tokens, roughly 2,048 px square at
 * their patch sizes, so images are reduced to MAX_IMAGE_EDGE on their long side. A few projectors
 * allow more and lose detail past it. The format an image arrived in is kept: PNG stays lossless,
 * which matters for screenshots and small text, and a JPEG was lossy already. Anything smaller, or
 * in a format that cannot be re-encoded here, is kept byte for byte.
 */

export const MAX_IMAGE_EDGE = 2048

export type ImageCopyPlan =
  | { action: 'copy' }
  | { action: 'resize'; width: number; height: number; format: 'png' | 'jpeg' }

export function planImageCopy(image: { width: number; height: number; ext: string }): ImageCopyPlan {
  const { width, height } = image
  if (!(width > 0 && height > 0)) return { action: 'copy' }
  const long = Math.max(width, height)
  if (long <= MAX_IMAGE_EDGE) return { action: 'copy' }

  const ext = image.ext.toLowerCase().replace(/^\./, '')
  const format = ext === 'jpg' || ext === 'jpeg' ? 'jpeg' : ext === 'png' ? 'png' : null
  if (!format) return { action: 'copy' }

  const scale = MAX_IMAGE_EDGE / long
  return {
    action: 'resize',
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
    format
  }
}
