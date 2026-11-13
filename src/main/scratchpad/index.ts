/**
 * The chat scratchpad, wired to the app: where it lives, how often it is swept, and what happens
 * to it when the app starts and quits. See store.ts for what it is and why.
 */

import os from 'node:os'
import { loadSettings } from '../storage/settings'
import { recordOf } from '../protect/keys'
import { logger } from '../log'
import { Scratchpad, clearOwnedFolder } from './store'
import { ScratchServer } from './serve'

/**
 * The chosen folder when the setting says to use one, otherwise null: memory. A protected
 * conversation's scratchpad is always in memory, so nothing of it is ever written out plain.
 */
function folder(chatId?: string): string | null {
  const s = loadSettings().scratchpad
  if (chatId && recordOf({ kind: 'chat', id: chatId })) return null
  return s?.location === 'folder' && s.folder ? s.folder : null
}

export const scratchpad = new Scratchpad({
  folder,
  /*
   * What a new file may take in memory: most of what the system has free, never all of it.
   *
   * Measured when the file arrives rather than fixed, so a machine with room to spare can hold a
   * long video and one that is already short of memory is told to use a folder instead of being
   * pushed into paging.
   */
  memoryLimit: () => Math.max(64 * 1024 ** 2, Math.floor(os.freemem() * 0.6))
})

export const scratchServer = new ScratchServer(scratchpad)

export function idleMinutes(): number {
  return Math.max(1, Math.round(loadSettings().scratchpad?.idleMinutes ?? 30))
}

type Listener = () => void
const listeners = new Set<Listener>()

/** Called whenever the scratchpad's contents change, so the window can show what it holds. */
export function onScratchpadChange(fn: Listener): () => void {
  listeners.add(fn)
  return () => listeners.delete(fn)
}

export function scratchpadChanged(): void {
  // The loopback server has nothing to serve once memory is empty.
  if (!scratchpad.totals().inMemory) scratchServer.stop()
  for (const fn of listeners) fn()
}

let timer: NodeJS.Timeout | null = null

/** Begin: clear what a crash may have left in the folder, then sweep idle files every half minute. */
export async function startScratchpad(): Promise<void> {
  await clearOwnedFolder(folder())
  if (timer) return
  timer = setInterval(() => {
    void scratchpad
      .sweep(idleMinutes() * 60_000)
      .then((gone) => {
        if (!gone.length) return
        logger.info('scratchpad', 'cleared files no longer in use', { files: gone.map((g) => g.name) })
        scratchpadChanged()
      })
      .catch((err: unknown) => logger.warn('scratchpad', 'sweep failed', { error: String(err) }))
  }, 30_000)
  timer.unref()
}

/** End: release memory and empty the folder, so nothing outlives the app. */
export async function stopScratchpad(): Promise<void> {
  if (timer) clearInterval(timer)
  timer = null
  scratchServer.stop()
  await scratchpad.clearAll().catch(() => undefined)
}
