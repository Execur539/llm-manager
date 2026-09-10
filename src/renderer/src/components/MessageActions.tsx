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
  editTitle = 'Edit'
}: {
  onCopy: () => void
  /** Absent where there is nothing sensible to edit — a tool step, or an empty turn. */
  onEdit?: () => void
  /** Says why editing is unavailable, when it is. */
  editTitle?: string
}): JSX.Element {
  const [copied, setCopied] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout>>()

  // Copying is silent and instant, so the tick is the only sign it worked.
  useEffect(() => () => clearTimeout(timer.current), [])

  const copy = (): void => {
    onCopy()
    setCopied(true)
    clearTimeout(timer.current)
    timer.current = setTimeout(() => setCopied(false), 1400)
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
    </div>
  )
}
