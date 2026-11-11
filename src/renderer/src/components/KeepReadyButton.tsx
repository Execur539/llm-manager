/**
 * The pin for a conversation's saved state.
 *
 * When conversations are kept ready on disk, the folder has a budget and the least recently used
 * ones go first. A pinned conversation is never removed to make room, so the one you always come
 * back to is always a file read away. Shown only while the feature is switched on.
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
          ? 'This conversation is kept ready on disk whatever the disk budget says. Click to let it be removed like the others.'
          : 'Never remove what the model has read of this conversation from disk to make room for others.'
      }
      aria-pressed={pinned}
      data-testid="keep-ready"
    >
      {pinned ? 'Kept ready' : 'Keep ready'}
    </button>
  )
}
