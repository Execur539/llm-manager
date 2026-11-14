/**
 * Saving a conversation's model state, by hand, and showing which of it is saved.
 *
 * Nothing saves a conversation's state on its own: the Save state button in the toolbar does, when
 * pressed, mid-reply included (the reply pauses, is saved, and carries on). Each message then
 * carries an edge: green for what the save holds, grey for what it does not, and for a reply saved
 * part way, green down to the exact point the save reached. While a save runs, the button fills
 * with its progress and the green creeps down the messages with it.
 *
 * Loading a conversation later loads what was saved and reads whatever came after it on top, so
 * nothing said after a save is lost; it is only read again.
 */

import { useEffect, useLayoutEffect, useState } from 'react'
import type { AgentMessage } from '@shared/types'
import { fmtBytes, invoke, on } from '../lib/api'
import { toast, useStream } from '../lib/store'
import Icon from './Icon'

/** Marks the exact point a save reached inside a reply's text, so the edge can be split there. */
export const SAVE_POINT = '⁤'

/** Which messages a conversation's save holds, as the main process works it out. */
export interface SaveCoverage {
  /** The last message the save holds whole, or null for none. */
  through: string | null
  /** A reply saved part way: which (null for the one still being written) and how far, from its first non-space character. */
  partial?: { id: string | null; content: number; reasoning: number } | null
  /** Saved under another model or other load settings: kept, but holding nothing while this one is loaded. */
  otherModel?: boolean
  tokens: number
  savedAt: number
}

export type SaveMark =
  | { state: 'saved' }
  | { state: 'unsaved' }
  | { state: 'saving' }
  | { state: 'partial'; content: number; reasoning: number }

/** What a save holds of a conversation, kept current as messages change and saves land. */
export function useSaveCoverage(stateKey: string | null, messages: AgentMessage[]): SaveCoverage | null {
  const stream = useStream()
  const [coverage, setCoverage] = useState<SaveCoverage | null>(null)
  // A model loaded or unloaded changes which saves can be used.
  const [modelTick, setModelTick] = useState(0)
  useEffect(() => on('model:status', () => setModelTick((n) => n + 1)), [])
  const tick = stateKey ? (stream.stateTicks[stateKey] ?? 0) : 0
  // What the messages say, compactly: an edit, a rewind or a new message changes it.
  const signature = messages.map((m) => `${m.id}:${m.content.length}:${m.reasoning?.length ?? 0}`).join('|')

  useEffect(() => {
    if (!stateKey) {
      setCoverage(null)
      return
    }
    let live = true
    const timer = setTimeout(() => {
      void invoke<SaveCoverage | null>('state:coverage', stateKey)
        .then((c) => {
          if (live) setCoverage(c)
        })
        .catch(() => undefined)
    }, 200)
    return () => {
      live = false
      clearTimeout(timer)
    }
  }, [stateKey, tick, signature, modelTick])
  return coverage
}

/** Characters of leading white space: offsets are counted from the first character that is not. */
const lead = (text: string): number => text.length - text.trimStart().length

/** `text` with the save point marked at `offset` characters past its leading white space. */
export function markSavePoint(text: string, offset: number): string {
  const at = Math.min(text.length, lead(text) + offset)
  return text.slice(0, at) + SAVE_POINT + text.slice(at)
}

/**
 * The edge each message carries: saved, not, or saved part way. While a save runs, the saved
 * colour creeps down the messages in step with its progress (by their share of the text, which is
 * near enough to the share of what is being written).
 */
export function saveMarks(
  messages: AgentMessage[],
  coverage: SaveCoverage | null,
  saving: { percent: number | null; stage: string } | undefined,
  streaming: { content: string; reasoning: string } | null
): { byId: Map<string, SaveMark>; streaming: SaveMark | null } {
  const byId = new Map<string, SaveMark>()
  const cover = coverage && !coverage.otherModel ? coverage : null
  const through = cover?.through ? messages.findIndex((m) => m.id === cover.through) : -1

  if (saving && saving.stage !== 'waiting') {
    if (saving.percent === null) {
      for (const m of messages) byId.set(m.id, { state: 'saving' })
      return { byId, streaming: streaming ? { state: 'saving' } : null }
    }
    const sizes = messages.map((m) => m.content.length + (m.reasoning?.length ?? 0) + 40)
    const tail = streaming ? streaming.content.length + streaming.reasoning.length + 40 : 0
    const total = sizes.reduce((a, b) => a + b, 0) + tail
    let sum = 0
    messages.forEach((m, i) => {
      sum += sizes[i]
      byId.set(m.id, { state: (sum / total) * 100 <= saving.percent! ? 'saved' : 'saving' })
    })
    return { byId, streaming: streaming ? { state: saving.percent >= 100 ? 'saved' : 'saving' } : null }
  }

  messages.forEach((m, i) => {
    if (i <= through) byId.set(m.id, { state: 'saved' })
    else if (cover?.partial?.id && cover.partial.id === m.id) {
      byId.set(m.id, { state: 'partial', content: cover.partial.content, reasoning: cover.partial.reasoning })
    } else byId.set(m.id, { state: 'unsaved' })
  })
  const partialLive = cover?.partial && cover.partial.id === null
  return {
    byId,
    streaming: streaming
      ? partialLive
        ? { state: 'partial', content: cover!.partial!.content, reasoning: cover!.partial!.reasoning }
        : { state: 'unsaved' }
      : null
  }
}

/** A message's text and thinking with the save point marked in whichever one the save stopped in. */
export function withSavePoint(
  mark: SaveMark | undefined | null,
  content: string,
  reasoning: string | undefined
): { content: string; reasoning: string | undefined } {
  if (!mark || mark.state !== 'partial') return { content, reasoning }
  // Stopped in the answer: the thinking before it is all saved.
  if (mark.content > 0) return { content: markSavePoint(content, mark.content), reasoning }
  if (reasoning && mark.reasoning < reasoning.trimStart().length) {
    return { content, reasoning: markSavePoint(reasoning, mark.reasoning) }
  }
  return { content: markSavePoint(content, 0), reasoning }
}

/**
 * Where, inside a message, its edge turns from saved to not: the line the save point is on, or
 * when that is out of sight (thinking folded away), the bottom of the thinking it is in.
 */
export function useSaveSplit(ref: React.RefObject<HTMLElement | null>, mark: SaveMark | undefined | null): void {
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    if (!mark || mark.state !== 'partial') {
      el.style.removeProperty('--save-split')
      return
    }
    const top = el.getBoundingClientRect().top
    let y: number | null = null
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT)
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const i = node.textContent?.indexOf(SAVE_POINT) ?? -1
      if (i < 0) continue
      const range = document.createRange()
      range.setStart(node, i)
      range.setEnd(node, i + 1)
      const rect = range.getBoundingClientRect()
      if (rect.height > 0) y = rect.top + rect.height / 2 - top
      break
    }
    if (y === null) {
      const thinking = el.querySelector('[data-testid="thinking-block"]')
      y = thinking ? thinking.getBoundingClientRect().bottom - top : 0
    }
    el.style.setProperty('--save-split', `${Math.max(0, Math.round(y))}px`)
  })
}

type SaveResult = { ok: true; tokens: number; savedAt: number; already?: boolean } | { ok: false; why: string }

/** The toolbar's Save state button: saves now, and fills with the save's progress while it runs. */
export function SaveStateButton({
  stateKey,
  canSave,
  allSaved,
  hasMessages
}: {
  stateKey: string | null
  canSave: boolean
  /** Every message is held by the save, and nothing is being written. */
  allSaved: boolean
  hasMessages: boolean
}): JSX.Element | null {
  const stream = useStream()
  const saving = stateKey ? stream.savingStates[stateKey] : undefined
  const [asking, setAsking] = useState(false)
  const busy = asking || !!saving
  if (!stateKey || !hasMessages) return null

  const save = async (): Promise<void> => {
    if (busy) return
    const key = stateKey
    setAsking(true)
    try {
      const r = await invoke<SaveResult>('state:save', key)
      if (!r.ok) toast(r.why, 'info')
      else if (r.already) toast('Already saved: nothing has been said since the last save.', 'info')
    } catch (err) {
      toast(`Could not save: ${err instanceof Error ? err.message : String(err)}`, 'error')
    } finally {
      setAsking(false)
    }
  }

  let label: string
  if (saving?.stage === 'waiting') label = 'Waiting to save…'
  else if (saving?.stage === 'reading') label = saving.percent !== null ? `Reading ${saving.percent}%` : 'Reading…'
  else if (saving) {
    label = saving.percent !== null ? `Saving ${saving.percent}%` : saving.bytes ? `Saving ${fmtBytes(saving.bytes)}` : 'Saving…'
  } else if (asking) label = 'Saving…'
  else label = allSaved ? 'Saved' : 'Save state'

  const title = saving?.stage === 'waiting'
    ? 'A tool is running or a call is being written; the save happens as soon as the model writes again, or when the reply ends.'
    : busy
      ? 'Saving what the model has read of this conversation, so coming back to it after a restart or another conversation is a file read.'
      : allSaved
        ? 'Everything in this conversation is saved. Saving again only does anything once more has been said.'
        : 'Save what the model has read of this conversation: green edges show what is saved, grey what is not. Anything after the save is read again when you come back, never lost. Works mid-reply: the reply pauses while it saves.'

  return (
    <button
      className={`save-state-button${busy ? ' working' : ''}${allSaved && !busy ? ' done' : ''}`}
      style={saving?.percent != null ? ({ '--save-progress': `${saving.percent}%` } as React.CSSProperties) : undefined}
      disabled={!canSave && !busy}
      onClick={() => void save()}
      title={title}
      aria-busy={busy || undefined}
      data-testid="save-state"
    >
      <span className="save-state-fill" aria-hidden="true" />
      <Icon name={allSaved && !busy ? 'check' : 'download'} size={14} />
      <span className="save-state-label">{label}</span>
    </button>
  )
}
