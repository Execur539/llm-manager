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
        {/*
          * A prompt for a hard-blocked action only happens when hard blocks are off.
          *
          * The engine refuses these outright while the list is on — it never asks. So reaching
          * this dialog means the user has already been through the buried setting and its typed
          * confirmation, and the honest thing is to warn loudly and still let them decide. It
          * used to hide every Allow button here instead, which made the whole escape hatch inert:
          * turning hard blocks off changed nothing, because the only remaining answer was Deny.
          */}
        <h2>
          <span>{current.hardBlocked ? 'Dangerous action' : 'Approve action'}</span>
          <span className={`badge ${current.tier === 'execute' ? 'bad' : 'warn'}`}>{current.tier}</span>
        </h2>

        {current.hardBlocked && (
          <div className="card error-card">
            <span className="badge bad">normally blocked</span> {current.blockReason}
            <div className="faint tiny-note">
              This is on the hard-block list and would normally be refused outright. You have hard
              blocks disabled in Settings, so it is yours to approve — it is not reversible.
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

        {!current.hardBlocked && (
          <div className="actions-remember">
            <span className="faint">Remember this decision:</span>
            <button className="link" onClick={() => respond('allow-exact')}>
              this exact call
            </button>
            <span className="faint">·</span>
            <button className="link" onClick={() => respond('allow-tool')}>
              every {current.tool}
            </button>
          </div>
        )}

        <div className="actions">
          <button className="danger" onClick={() => respond('deny')} data-testid="permission-deny">
            Deny
          </button>
          {/* Once only, never remembered: an unrecoverable action should be decided every time. */}
          <button
            className={current.hardBlocked ? 'danger' : 'primary'}
            onClick={() => respond('allow-once')}
            data-testid="permission-allow"
          >
            {current.hardBlocked ? 'Allow anyway' : 'Allow once'}
          </button>
        </div>
      </div>
    </div>
  )
}
