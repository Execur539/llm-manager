/**
 * Whether what the model has read of this conversation is saved to disk.
 *
 * Saving happens on its own (a few seconds after a reply, before the computer sleeps, on switching
 * conversations), which also meant nothing on screen ever said it had. Shown beside the context
 * ring: "Saved" for a moment when a save lands, then a quiet tick that says, on hover, how much is
 * kept and when. Coming back to a conversation with that tick, after a restart or a model reload,
 * is a file read rather than a wait while the model reads it all again.
 */

import { useEffect, useState } from 'react'
import { invoke, fmtRelative } from '../lib/api'
import { useStream } from '../lib/store'
import Icon from './Icon'

export default function SavedState({ stateKey }: { stateKey: string | null }): JSX.Element | null {
  const stream = useStream()
  const [stored, setStored] = useState<{ tokens: number; savedAt: number } | null>(null)
  const [fresh, setFresh] = useState(false)
  const live = stateKey ? stream.savedStates[stateKey] : undefined

  // What is on disk already, for a conversation opened rather than just saved.
  useEffect(() => {
    setStored(null)
    if (!stateKey) return
    let current = true
    void invoke<{ tokens: number; savedAt: number } | null>('state:meta', stateKey)
      .then((m) => {
        if (current) setStored(m)
      })
      .catch(() => undefined)
    return () => {
      current = false
    }
  }, [stateKey])

  // A save that just landed says so in words for a couple of seconds, then settles to the tick.
  useEffect(() => {
    if (!live) return
    setFresh(true)
    const t = setTimeout(() => setFresh(false), 2400)
    return () => clearTimeout(t)
  }, [live?.savedAt])

  const saved = live && (!stored || live.savedAt >= stored.savedAt) ? live : stored
  if (!stateKey || !saved) return null

  return (
    <span
      className={`saved-state${fresh ? ' fresh' : ''}`}
      title={`Saved to disk ${fmtRelative(saved.savedAt)}: ${saved.tokens.toLocaleString()} tokens the model will not have to read again after a restart or a model reload.`}
      data-testid="saved-state"
    >
      <Icon name="check" size={12} />
      <span className="saved-state-text">{fresh ? 'Saved' : 'Ready'}</span>
    </span>
  )
}
