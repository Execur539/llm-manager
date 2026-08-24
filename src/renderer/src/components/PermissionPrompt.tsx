import { useEffect } from 'react'
import type { PermissionDecision } from '@shared/types'
import { send } from '../lib/api'
import { dismissPermission, useStream } from '../lib/store'

/**
 * The approval gate.
 *
 * Shows the fully-resolved action as computed in the main process — never the model's own
 * description of what it is doing — so intent injected through a file or web page has nowhere
 * to hide. Paths are canonicalised before they reach here.
 *
 * The queue lives in the store rather than in this component, so a prompt raised while the user
 * is on another page is still waiting when they return, instead of being lost with the unmount.
 */
export default function PermissionPrompt(): JSX.Element | null {
  const { permissionQueue } = useStream()
  const current = permissionQueue[0]

  // Keyboard handling: Escape denies. There is deliberately no Enter-to-approve, because a
  // stray keypress should never be able to authorise a write or a shell command.
  useEffect(() => {
    if (!current) return
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') {
        e.preventDefault()
        respond('deny')
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [current])

  if (!current) return null

  function respond(decision: PermissionDecision): void {
    if (!current) return
    send('agent:permission-response', current.id, decision)
    dismissPermission(current.id)
  }

  return (
    <div className="overlay" data-testid="permission-overlay">
      <div className="modal" role="dialog" aria-modal="true" aria-label="Approve action">
        <h2>
          <span>{current.hardBlocked ? 'Blocked action' : 'Approve action'}</span>
          <span className={`badge ${current.tier === 'execute' ? 'bad' : 'warn'}`}>{current.tier}</span>
        </h2>

        {current.hardBlocked && (
          <div className="card error-card">
            <span className="badge bad">hard-blocked</span> {current.blockReason}
            <div className="faint tiny-note">
              This cannot be approved here. Hard blocks are only disabled in Settings, behind a typed
              confirmation.
            </div>
          </div>
        )}

        <div className="faint modal-label">
          Tool: <span className="mono">{current.tool}</span>
        </div>
        <pre className="resolved-action">{current.resolved}</pre>

        {permissionQueue.length > 1 && (
          <div className="faint tiny-note">{permissionQueue.length - 1} more waiting</div>
        )}

        <div className="actions">
          <button className="danger" onClick={() => respond('deny')} data-testid="permission-deny">
            Deny
          </button>
          {!current.hardBlocked && (
            <>
              <button onClick={() => respond('allow-exact')}>Always allow this exact call</button>
              <button onClick={() => respond('allow-tool')}>Always allow {current.tool}</button>
              <button className="primary" onClick={() => respond('allow-once')} data-testid="permission-allow">
                Allow once
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
