import { useEffect, useRef, useState } from 'react'

/**
 * In-app confirmation, optionally requiring a typed phrase.
 *
 * Electron does not implement `window.prompt`, so a typed confirmation built on it throws
 * "prompt() is not supported" and the control it guards becomes unusable. Native `confirm` does
 * work, but it cannot ask for a phrase and looks nothing like the rest of the app.
 *
 * Escape and the backdrop both cancel. There is no Enter-to-confirm: these dialogs guard
 * destructive or safety-relevant actions, so confirming is always a deliberate click.
 */
export default function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  requirePhrase,
  danger = false,
  onConfirm,
  onCancel
}: {
  open: boolean
  title: string
  body: React.ReactNode
  confirmLabel?: string
  cancelLabel?: string
  /** When set, the confirm button stays disabled until this exact phrase is typed. */
  requirePhrase?: string
  danger?: boolean
  onConfirm: () => void
  onCancel: () => void
}): JSX.Element | null {
  const [typed, setTyped] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const confirmRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!open) {
      setTyped('')
      return
    }
    // Focus the field that actually gates the action.
    const target = requirePhrase ? inputRef.current : confirmRef.current
    target?.focus()

    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') {
        e.preventDefault()
        onCancel()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, requirePhrase, onCancel])

  if (!open) return null

  const satisfied = !requirePhrase || typed === requirePhrase

  return (
    <div className="overlay" onClick={onCancel} data-testid="confirm-overlay">
      <div
        className="modal confirm-modal"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(e) => e.stopPropagation()}
      >
        <h2>{title}</h2>
        <div className="confirm-body">{body}</div>

        {requirePhrase && (
          <label className="confirm-phrase">
            <span className="faint">
              Type <code className="mono">{requirePhrase}</code> to continue
            </span>
            <input
              ref={inputRef}
              type="text"
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              autoComplete="off"
              spellCheck={false}
              data-testid="confirm-phrase-input"
            />
          </label>
        )}

        <div className="actions">
          <button onClick={onCancel} data-testid="confirm-cancel">
            {cancelLabel}
          </button>
          <button
            ref={confirmRef}
            className={danger ? 'danger' : 'primary'}
            disabled={!satisfied}
            onClick={onConfirm}
            data-testid="confirm-accept"
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
