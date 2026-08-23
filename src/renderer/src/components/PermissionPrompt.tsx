import { useEffect, useState } from 'react'
import type { PermissionDecision, PermissionRequest } from '@shared/types'
import { on, send } from '../lib/api'

/**
 * The approval gate.
 *
 * Shows the fully-resolved action as computed in the main process — never the model's own
 * description of what it is doing — so intent injected through a file or web page has nowhere
 * to hide. Paths are canonicalised before they reach here.
 */
export default function PermissionPrompt(): JSX.Element | null {
  const [queue, setQueue] = useState<PermissionRequest[]>([])

  useEffect(() => on<PermissionRequest>('agent:permission-request', (r) => setQueue((q) => [...q, r])), [])

  const current = queue[0]
  if (!current) return null

  const respond = (decision: PermissionDecision): void => {
    send('agent:permission-response', current.id, decision)
    setQueue((q) => q.slice(1))
  }

  return (
    <div className="overlay">
      <div className="modal">
        <h2>
          {current.hardBlocked ? 'Blocked action' : 'Approve action'}
          <span className={`badge ${current.tier === 'execute' ? 'bad' : 'warn'}`}>{current.tier}</span>
        </h2>

        {current.hardBlocked && (
          <div className="card" style={{ borderColor: '#5c2626' }}>
            <span className="badge bad">hard-blocked</span> {current.blockReason}
            <div className="faint" style={{ marginTop: 6 }}>
              This cannot be approved here. Hard blocks are only disabled in Settings, behind a typed confirmation.
            </div>
          </div>
        )}

        <div className="faint" style={{ marginBottom: 4 }}>
          Tool: <span className="mono">{current.tool}</span>
        </div>
        <pre>{current.resolved}</pre>

        {queue.length > 1 && (
          <div className="faint" style={{ fontSize: 11 }}>{queue.length - 1} more waiting</div>
        )}

        <div className="actions">
          <button className="danger" onClick={() => respond('deny')}>Deny</button>
          {!current.hardBlocked && (
            <>
              <button onClick={() => respond('allow-exact')}>Always allow this exact call</button>
              <button onClick={() => respond('allow-tool')}>Always allow {current.tool}</button>
              <button className="primary" onClick={() => respond('allow-once')}>Allow once</button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
