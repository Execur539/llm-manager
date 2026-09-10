/**
 * What you can do to a turn, out of the way until you want it.
 *
 * Floated over the turn rather than placed in its flow, for the same reason the conversation
 * rail's actions are: anything that takes space on hover moves every message below it, and a
 * transcript that shifts as the pointer crosses it is unreadable.
 */

import { useEffect, useRef, useState } from 'react'
import Icon from './Icon'

export default function MessageActions({
  onCopy,
  onEdit,
  editTitle = 'Edit',
  onDelete,
  deleteTitle = 'Delete this and everything after it'
}: {
  onCopy: () => void
  /** Absent where there is nothing sensible to edit — a tool step, or an empty turn. */
  onEdit?: () => void
  /** Says why editing is unavailable, when it is. */
  editTitle?: string
  /** Absent where deleting makes no sense on its own — a tool result belongs to its call. */
  onDelete?: () => void
  deleteTitle?: string
}): JSX.Element {
  const [copied, setCopied] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout>>()
  // A second click is needed to actually remove anything: this takes the rest of the
  // conversation with it, and that is not something a stray click should be able to do.
  const [confirming, setConfirming] = useState(false)

  // Copying is silent and instant, so the tick is the only sign it worked.
  useEffect(() => () => clearTimeout(timer.current), [])
  // The row is re-hidden on mouseleave by the hover CSS; if it is shown again later for a
  // different reason, a stale confirm step should not still be sitting there armed.
  useEffect(() => () => setConfirming(false), [])

  const copy = (): void => {
    onCopy()
    setCopied(true)
    clearTimeout(timer.current)
    timer.current = setTimeout(() => setCopied(false), 1400)
  }

  if (confirming) {
    return (
      <div className="msg-actions confirming" data-testid="message-actions">
        <button
          className="row-action danger"
          onClick={() => {
            setConfirming(false)
            onDelete?.()
          }}
          title="Confirm delete"
          aria-label="Confirm deleting this message and everything after it"
          data-testid="message-delete-confirm"
        >
          <Icon name="check" size={13} />
        </button>
        <button
          className="row-action"
          onClick={() => setConfirming(false)}
          title="Cancel"
          aria-label="Cancel delete"
          data-testid="message-delete-cancel"
        >
          <Icon name="close" size={13} />
        </button>
      </div>
    )
  }

  return (
    <div className="msg-actions" data-testid="message-actions">
      <button
        className="row-action"
        onClick={copy}
        title={copied ? 'Copied' : 'Copy'}
        aria-label="Copy message"
        data-testid="message-copy"
      >
        <Icon name={copied ? 'check' : 'copy'} size={13} />
      </button>
      {onEdit && (
        <button
          className="row-action"
          onClick={onEdit}
          title={editTitle}
          aria-label="Edit message"
          data-testid="message-edit"
        >
          <Icon name="pencil" size={13} />
        </button>
      )}
      {onDelete && (
        <button
          className="row-action danger"
          onClick={() => setConfirming(true)}
          title={deleteTitle}
          aria-label="Delete this message and everything after it"
          data-testid="message-delete"
        >
          <Icon name="trash" size={13} />
        </button>
      )}
    </div>
  )
}
