/**
 * The app's own copies of what was attached to a conversation.
 *
 * A transcript used to record where the user's file was, not the file. Moving, renaming or deleting
 * the original took the image out of the conversation after a restart; editing it changed what the
 * model saw; and pasted images, which only ever lived in the tool-output folder, were swept away
 * with it after a week — as were the sampled video clips the transcript plays.
 *
 * Images and audio are now copied in when a message is sent, images reduced to what a projector can
 * use (see image-copy), and that copy is what the model is sent: on the first turn, and again on any
 * rebuilt history, so the pixels never differ between the two. Video sources stay where they are —
 * copying a film to keep a few frames of it would be absurd — but the clip and stills made from one
 * are moved in beside the rest. Copies live in one folder per conversation, go when it does, and are
 * never aged out.
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import crypto from 'node:crypto'
import { nativeImage } from 'electron'
import { ATTACHMENTS_DIR } from '../storage/paths'
import { logger } from '../log'
import { allAttachmentPaths, attachmentRowsForChat, classifyAttachment, setAttachmentStored } from './repo'
import { planImageCopy } from './image-copy'

const segment = (s: string): string => s.replace(/[^A-Za-z0-9_-]/g, '_')

/** Whether a path is one of this app's kept copies. */
export function isStoredCopy(file: string): boolean {
  const rel = path.relative(ATTACHMENTS_DIR, file)
  return !!rel && !rel.startsWith('..') && !path.isAbsolute(rel)
}

/** A fresh folder for one attachment, so the file inside can keep its own name without colliding. */
async function newSlot(chatId: string): Promise<string> {
  const dir = path.join(ATTACHMENTS_DIR, segment(chatId), crypto.randomBytes(6).toString('hex'))
  await fsp.mkdir(dir, { recursive: true })
  return dir
}

/** Writes a reduced copy when the image needs one; false when a plain copy will do. */
async function writeReducedImage(file: string, dest: string): Promise<boolean> {
  const image = nativeImage.createFromPath(file)
  // Empty for a format Electron cannot decode, which is then kept exactly as it came.
  if (image.isEmpty()) return false
  const { width, height } = image.getSize()
  const plan = planImageCopy({ width, height, ext: path.extname(file) })
  if (plan.action === 'copy') return false
  const reduced = image.resize({ width: plan.width, height: plan.height, quality: 'best' })
  await fsp.writeFile(dest, plan.format === 'jpeg' ? reduced.toJPEG(90) : reduced.toPNG())
  return true
}

/**
 * Keep a copy of an attached image or audio file, returning the path the message should use.
 *
 * Video and documents are left where they are and their own path returned, as is a file that
 * cannot be copied — a message is never failed over keeping a copy of its attachment.
 */
export async function storeAttachment(chatId: string, file: string): Promise<string> {
  const kind = classifyAttachment(file)
  if ((kind !== 'image' && kind !== 'audio') || isStoredCopy(file)) return file
  try {
    const dest = path.join(await newSlot(chatId), path.basename(file))
    if (kind === 'image' && (await writeReducedImage(file, dest))) return dest
    await fsp.copyFile(file, dest)
    return dest
  } catch (err) {
    logger.warn('attachments', `could not keep a copy of ${file}; the original is referenced instead`, err)
    return file
  }
}

async function moveFile(from: string, to: string): Promise<void> {
  try {
    await fsp.rename(from, to)
  } catch {
    // Across volumes a rename cannot work; a copy and a delete do the same job.
    await fsp.copyFile(from, to)
    await fsp.rm(from, { force: true })
  }
}

/**
 * Move the clip and stills made from a video out of the folder that is swept every week.
 *
 * Moved rather than copied: they were made for this message alone, and nothing else refers to them.
 */
export async function keepPrepared(
  chatId: string,
  media: { optimised?: string; stills?: string[] }
): Promise<{ optimised?: string; stills?: string[] }> {
  const pending = [media.optimised, ...(media.stills ?? [])].filter(
    (f): f is string => !!f && !isStoredCopy(f) && fs.existsSync(f)
  )
  if (!pending.length) return media
  const dir = await newSlot(chatId)
  const moved = new Map<string, string>()
  for (const [i, file] of pending.entries()) {
    const dest = path.join(dir, `${i}-${path.basename(file)}`)
    try {
      await moveFile(file, dest)
      moved.set(file, dest)
    } catch (err) {
      logger.warn('attachments', `could not keep ${file}`, err)
    }
  }
  return {
    optimised: media.optimised ? (moved.get(media.optimised) ?? media.optimised) : undefined,
    stills: media.stills?.map((s) => moved.get(s) ?? s)
  }
}

/**
 * Bring a conversation's older attachments into the app's own storage.
 *
 * Messages sent before copies were kept point at the user's files, and at clips in the folder that
 * is swept weekly. Each one still there is copied or moved in, once, the next time the conversation
 * is opened or continued. One already gone is left as it is: there is nothing left to keep.
 */
export async function adoptAttachments(chatId: string): Promise<void> {
  for (const row of attachmentRowsForChat(chatId)) {
    try {
      let file = row.path
      if ((row.kind === 'image' || row.kind === 'audio') && !isStoredCopy(file) && fs.existsSync(file)) {
        file = await storeAttachment(chatId, file)
      }
      let meta = row.meta
      if (row.kind === 'video' && (meta.optimised || meta.stills?.length)) {
        const kept = await keepPrepared(chatId, meta)
        if (kept.optimised !== meta.optimised || kept.stills?.some((s, i) => s !== meta.stills?.[i])) {
          meta = { ...meta, optimised: kept.optimised, stills: kept.stills }
        }
      }
      if (file !== row.path) meta = { ...meta, original: meta.original ?? row.path }
      if (file !== row.path || meta !== row.meta) setAttachmentStored(row.id, file, meta)
    } catch (err) {
      logger.warn('attachments', `could not keep the files of attachment ${row.id}`, err)
    }
  }
}

/** Delete a conversation's kept copies, when the conversation itself is deleted. */
export async function deleteConversationFiles(chatId: string): Promise<void> {
  await fsp.rm(path.join(ATTACHMENTS_DIR, segment(chatId)), { recursive: true, force: true }).catch(() => undefined)
}

/**
 * Remove kept copies no message refers to any more.
 *
 * They are left behind when a message is deleted or a conversation rewound, since the database
 * drops the rows on its own. Only folders more than an hour old are touched, so a message still
 * being sent never loses a copy it has not recorded yet.
 */
export async function pruneOrphanAttachments(): Promise<void> {
  const referenced = new Set<string>()
  for (const file of allAttachmentPaths()) {
    if (!isStoredCopy(file)) continue
    const [chat, slot] = path.relative(ATTACHMENTS_DIR, file).split(path.sep)
    if (chat && slot) referenced.add(path.join(ATTACHMENTS_DIR, chat, slot))
  }

  const cutoff = Date.now() - 60 * 60 * 1000
  for (const chat of await fsp.readdir(ATTACHMENTS_DIR, { withFileTypes: true }).catch(() => [])) {
    if (!chat.isDirectory()) continue
    const chatDir = path.join(ATTACHMENTS_DIR, chat.name)
    for (const slot of await fsp.readdir(chatDir, { withFileTypes: true }).catch(() => [])) {
      const slotDir = path.join(chatDir, slot.name)
      if (referenced.has(slotDir)) continue
      try {
        if ((await fsp.stat(slotDir)).mtimeMs < cutoff) await fsp.rm(slotDir, { recursive: true, force: true })
      } catch {
        // Best effort: anything that cannot be examined is left alone.
      }
    }
    const left = await fsp.readdir(chatDir).catch(() => null)
    if (left && !left.length) await fsp.rm(chatDir, { recursive: true, force: true }).catch(() => undefined)
  }
}
