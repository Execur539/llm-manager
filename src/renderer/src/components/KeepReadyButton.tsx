/**
 * The pin for a conversation's saved state.
 *
 * Every conversation is saved to disk as it goes (a few seconds after each reply), and the folder
 * has a budget: the least recently used go first. A pinned conversation is never removed to make
 * room, so the one you always come back to is always a file read away. Shown only while keeping
 * conversations ready is switched on, which it is by default.
 */

import { useEffect, useState } from 'react'
import { invoke } from '../lib/api'

interface StateInfo {
  enabled: boolean
  pinned: string[]
}

export default function KeepReadyButton({ stateKey }: { stateKey: string | null }): JSX.Element | null {
  const [info, setInfo] = useState<StateInfo | null>(null)

  useEffect(() => {
    let live = true
    void invoke<StateInfo>('state:info')
      .then((i) => {
        if (live) setInfo(i)
      })
      .catch(() => undefined)
    return () => {
      live = false
    }
  }, [stateKey])

  if (!info?.enabled || !stateKey) return null
  const pinned = info.pinned.includes(stateKey)

  return (
    <button
      className={pinned ? 'active' : undefined}
      onClick={() => {
        void invoke<string[]>('state:pin', stateKey, !pinned)
          .then((pins) => setInfo({ ...info, pinned: pins }))
          .catch(() => undefined)
      }}
      title={
        pinned
          ? 'Kept ready on disk whatever the disk budget says. Click to let it be removed to make room, like the others.'
          : 'Every conversation is saved to disk as you go, and the least recently used are removed when the disk budget fills. Click to never remove this one.'
      }
      aria-pressed={pinned}
      data-testid="keep-ready"
    >
      {pinned ? 'Kept ready' : 'Keep ready'}
    </button>
  )
}
