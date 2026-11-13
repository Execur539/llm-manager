/**
 * Whether what the model has read of this conversation is saved to disk, and a way to save it now.
 *
 * Saving happens on its own (a few seconds after a reply, before the computer sleeps, on switching
 * conversations), which also meant nothing on screen ever said it had. Shown beside the context
 * ring: "Saved" for a moment when a save lands, then a quiet tick that says, on hover, how much is
 * kept and when. Coming back to a conversation with that tick, after a restart or a model reload,
 * is a file read rather than a wait while the model reads it all again.
 *
 * The app does not save everything on its own: not a short conversation, and not one with little
 * new since its last save. Clicking saves it now regardless: "Save state" when nothing is on disk
 * yet, the tick to save again. Reading a conversation the model is not holding comes first, with
 * how far it has got.
 */

import { useEffect, useState } from 'react'
import { invoke, fmtRelative } from '../lib/api'
import { toast, useStream } from '../lib/store'
import Icon from './Icon'
import { Spinner } from './Spinner'

type SaveResult = { ok: true; tokens: number; savedAt: number; already?: boolean } | { ok: false; why: string }

export default function SavedState({
  stateKey,
  canSave
}: {
  stateKey: string | null
  /** A model is loaded, nothing is being written, and the conversation has something in it. */
  canSave: boolean
}): JSX.Element | null {
  const stream = useStream()
  const [stored, setStored] = useState<{ tokens: number; savedAt: number } | null>(null)
  const [enabled, setEnabled] = useState(false)
  const [flash, setFlash] = useState<'saved' | 'current' | null>(null)
  const [asking, setAsking] = useState(false)
  const live = stateKey ? stream.savedStates[stateKey] : undefined
  const saving = stateKey ? stream.savingStates[stateKey] : undefined

  // What is on disk already, for a conversation opened rather than just saved, and whether
  // conversations are kept at all.
  useEffect(() => {
    setStored(null)
    setFlash(null)
    if (!stateKey) return
    let current = true
    void invoke<{ tokens: number; savedAt: number } | null>('state:meta', stateKey)
      .then((m) => {
        if (current) setStored(m)
      })
      .catch(() => undefined)
    void invoke<{ enabled: boolean }>('state:info')
      .then((i) => {
        if (current) setEnabled(!!i.enabled)
      })
      .catch(() => undefined)
    return () => {
      current = false
    }
  }, [stateKey])

  // A save that just landed says so in words for a couple of seconds, then settles to the tick.
  useEffect(() => {
    if (live) setFlash('saved')
  }, [live?.savedAt])

  useEffect(() => {
    if (!flash) return
    const t = setTimeout(() => setFlash(null), 2400)
    return () => clearTimeout(t)
  }, [flash])

  const saved = live && (!stored || live.savedAt >= stored.savedAt) ? live : stored
  const working = asking || !!saving
  if (!stateKey || !enabled) return null
  // Nothing on disk and nothing to be done about it right now: nothing to show.
  if (!saved && !working && !canSave) return null

  const save = async (): Promise<void> => {
    if (working) return
    setAsking(true)
    try {
      const r = await invoke<SaveResult>('state:save', stateKey)
      if (!r.ok) toast(r.why, 'info')
      else if (r.already) setFlash('current')
      // A save that landed is announced by state:saved, and flashes "Saved" above.
    } catch (err) {
      toast(`Could not save: ${err instanceof Error ? err.message : String(err)}`, 'error')
    } finally {
      setAsking(false)
    }
  }

  let label: string
  if (saving?.phase === 'reading') {
    label = saving.total ? `Reading ${Math.round(((saving.done ?? 0) / saving.total) * 100)}%` : 'Reading…'
  } else if (saving?.phase === 'saving') label = 'Saving…'
  else if (asking) label = 'Checking…'
  else if (flash === 'saved') label = 'Saved'
  else if (flash === 'current') label = 'Up to date'
  else if (saved) label = 'Ready'
  else label = 'Save state'

  const title = working
    ? saving?.phase === 'reading'
      ? 'The model is reading this conversation in, so that what it has read can be saved.'
      : 'Saving what the model has read of this conversation to disk.'
    : saved
      ? `Saved to disk ${fmtRelative(saved.savedAt)}: ${saved.tokens.toLocaleString()} tokens the model will not have to read again after a restart or a model reload.${canSave ? ' Click to save it again now.' : ''}`
      : 'What the model has read of this conversation is not on disk yet: the app saves on its own once a conversation is long enough to be worth it. Click to save it now anyway.'

  return (
    <button
      type="button"
      className={`saved-state${flash ? ' fresh' : ''}${!saved && !working ? ' unsaved' : ''}`}
      onClick={() => void save()}
      disabled={!canSave && !working}
      aria-busy={working || undefined}
      title={title}
      data-testid="saved-state"
    >
      {working ? <Spinner size={10} /> : <Icon name={saved || flash ? 'check' : 'download'} size={12} />}
      <span className="saved-state-text">{label}</span>
    </button>
  )
}
