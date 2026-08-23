/**
 * Turn checkpoints.
 *
 * Before a write-class tool runs, every file it is about to touch is snapshotted. That makes
 * "rewind to before this message" possible without requiring the workspace to be a git repo —
 * and since scope is machine-wide, most of what the agent touches will not be one.
 *
 * Deleted-file handling matters: a checkpoint records that a file did NOT exist so a rewind
 * can remove one the agent created.
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import crypto from 'node:crypto'
import { CHECKPOINTS_DIR } from '../storage/paths'

interface CheckpointEntry {
  originalPath: string
  /** null when the file did not exist at checkpoint time */
  storedAs: string | null
  bytes: number
}

interface Checkpoint {
  id: string
  sessionId: string
  createdAt: number
  entries: CheckpointEntry[]
}

function manifestPath(sessionId: string): string {
  return path.join(CHECKPOINTS_DIR, `${sessionId}.json`)
}

async function readManifest(sessionId: string): Promise<Checkpoint[]> {
  try {
    return JSON.parse(await fsp.readFile(manifestPath(sessionId), 'utf8')) as Checkpoint[]
  } catch {
    return []
  }
}

async function writeManifest(sessionId: string, checkpoints: Checkpoint[]): Promise<void> {
  await fsp.mkdir(CHECKPOINTS_DIR, { recursive: true })
  await fsp.writeFile(manifestPath(sessionId), JSON.stringify(checkpoints, null, 2))
}

/** Snapshot the given paths. Returns the checkpoint id. */
export async function checkpointFiles(sessionId: string, paths: string[]): Promise<string | null> {
  if (!paths.length) return null

  const id = crypto.randomBytes(6).toString('hex')
  const dir = path.join(CHECKPOINTS_DIR, sessionId, id)
  await fsp.mkdir(dir, { recursive: true })

  const entries: CheckpointEntry[] = []
  for (const p of paths) {
    const abs = path.resolve(p)
    try {
      const st = await fsp.stat(abs)
      if (!st.isFile()) continue
      // Don't try to snapshot something enormous; record it as unsnapshotted instead.
      if (st.size > 256 * 1024 * 1024) {
        entries.push({ originalPath: abs, storedAs: null, bytes: st.size })
        continue
      }
      const stored = path.join(dir, `${crypto.randomBytes(4).toString('hex')}-${path.basename(abs)}`)
      await fsp.copyFile(abs, stored)
      entries.push({ originalPath: abs, storedAs: stored, bytes: st.size })
    } catch {
      // The file does not exist yet: record that, so a rewind deletes whatever gets created.
      entries.push({ originalPath: abs, storedAs: null, bytes: -1 })
    }
  }

  const checkpoints = await readManifest(sessionId)
  checkpoints.push({ id, sessionId, createdAt: Date.now(), entries })
  await writeManifest(sessionId, checkpoints)
  return id
}

/** Restore everything captured at or after the given checkpoint, newest first. */
export async function rewindTo(sessionId: string, checkpointId: string): Promise<{ restored: number; removed: number }> {
  const checkpoints = await readManifest(sessionId)
  const index = checkpoints.findIndex((c) => c.id === checkpointId)
  if (index < 0) throw new Error(`No checkpoint ${checkpointId}`)

  let restored = 0
  let removed = 0

  for (let i = checkpoints.length - 1; i >= index; i--) {
    for (const entry of checkpoints[i].entries) {
      if (entry.storedAs && fs.existsSync(entry.storedAs)) {
        await fsp.mkdir(path.dirname(entry.originalPath), { recursive: true })
        await fsp.copyFile(entry.storedAs, entry.originalPath)
        restored++
      } else if (entry.bytes === -1) {
        // Did not exist before: remove whatever the agent created.
        try {
          await fsp.rm(entry.originalPath, { force: true })
          removed++
        } catch {
          /* best effort */
        }
      }
    }
  }

  await writeManifest(sessionId, checkpoints.slice(0, index))
  return { restored, removed }
}

export async function listCheckpoints(sessionId: string): Promise<{ id: string; createdAt: number; files: string[] }[]> {
  const checkpoints = await readManifest(sessionId)
  return checkpoints.map((c) => ({
    id: c.id,
    createdAt: c.createdAt,
    files: c.entries.map((e) => e.originalPath)
  }))
}
