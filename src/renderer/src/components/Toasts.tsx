/**
 * Transient confirmations, stacked bottom-right.
 *
 * Rendered once at the app root rather than per view, so a toast raised by an action survives
 * navigating away from the view that raised it — exporting a conversation and then switching
 * to Dashboard still tells you where the file went.
 */

import { dismissToast, useStream } from '../lib/store'
import { invoke } from '../lib/api'

export default function Toasts(): JSX.Element | null {
  const { toasts } = useStream()
  if (toasts.length === 0) return null

  return (
    <div className="toast-stack" role="status" aria-live="polite" data-testid="toast-stack">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`} data-testid="toast">
          <span className="toast-text">{t.message}</span>
          {t.revealPath && (
            <button
              className="link"
              onClick={() => void invoke('shell:reveal', t.revealPath)}
              data-testid="toast-reveal"
            >
              Show
            </button>
          )}
          <button
            className="toast-close"
            onClick={() => dismissToast(t.id)}
            aria-label="Dismiss"
            title="Dismiss"
            data-testid="toast-dismiss"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  )
}
