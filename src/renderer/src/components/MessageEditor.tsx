/**
 * Rewriting a turn that has already been said.
 *
 * A reasoning model's turn is two texts, not one, and they read very differently: the chain of
 * thought is the model talking to itself, the reply is it talking to you. Editing them in a
 * single box would make it guesswork which half of the text you were changing, so they get one
 * editor each — the thinking one carrying the same inset, dimmed treatment the collapsed
 * ThinkingBlock uses, so the two are recognisable at a glance without reading a word of them.
 *
 * Turns with no reasoning show a single editor and no labels, because with nothing to
 * distinguish it from, a label is just a word taking up room.
 */

import { useEffect, useRef, useState } from 'react'
import Icon from './Icon'

/** Grows with its content, up to a point, so a long reply is not edited through a slot. */
function useAutoGrow(value: string): React.RefObject<HTMLTextAreaElement> {
  const ref = useRef<HTMLTextAreaElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    /*
     * scrollHeight covers content and padding but not the border, and the box is sized
     * border-box — so setting the height to it alone leaves the box two pixels short of its own
     * text and every editor opens with a scrollbar it does not need.
     */
    const style = getComputedStyle(el)
    const border = parseFloat(style.borderTopWidth) + parseFloat(style.borderBottomWidth)
    el.style.height = `${Math.min(el.scrollHeight + border, 420)}px`
  }, [value])
  return ref
}

export default function MessageEditor({
  content,
  reasoning,
  onSave,
  onCancel,
  disabled,
  disabledReason
}: {
  content: string
  /** Absent when the turn had no chain of thought — then there is only one thing to edit. */
  reasoning?: string
  onSave: (content: string, reasoning?: string) => void
  onCancel: () => void
  /*
   * Saving can no longer be answered — a turn started elsewhere in this session while the
   * editor sat open, or nothing is loaded to answer with. `saving` alone does not cover this:
   * that flag only guards against a second click on this same button, not against the ground
   * having shifted under an editor that was opened before either became true.
   */
  disabled?: boolean
  /** Shown on the Save button when `disabled` is set, so the reason is not just a dead click. */
  disabledReason?: string
}): JSX.Element {
  const [body, setBody] = useState(content)
  const [thought, setThought] = useState(reasoning ?? '')
  const [saving, setSaving] = useState(false)

  const bodyRef = useAutoGrow(body)
  const thoughtRef = useAutoGrow(thought)

  // The reply is what people came to change, so that is where the cursor starts.
  useEffect(() => {
    const el = bodyRef.current
    if (!el) return
    el.focus()
    el.setSelectionRange(el.value.length, el.value.length)
  }, [])

  const save = (): void => {
    if (saving || disabled) return
    setSaving(true)
    onSave(body, reasoning === undefined ? undefined : thought)
  }

  /*
   * Escape cancels, Ctrl+Enter saves. Plain Enter has to stay a newline: unlike the composer,
   * where a message is usually one line, everything edited here is already prose with line
   * breaks in it, and losing them to an accidental save would be worse than the extra key.
   */
  const keys = (e: React.KeyboardEvent): void => {
    if (e.key === 'Escape') {
      e.preventDefault()
      onCancel()
    } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      save()
    }
  }

  return (
    <div className="msg-editor" data-testid="message-editor">
      {reasoning !== undefined && (
        <div className="msg-editor-pane cot">
          <div className="msg-editor-label">
            <Icon name="sparkle" size={11} />
            <span>Thought process</span>
            <span className="faint">the model's own working, not part of its reply</span>
          </div>
          <textarea
            ref={thoughtRef}
            value={thought}
            onChange={(e) => setThought(e.target.value)}
            onKeyDown={keys}
            spellCheck={false}
            data-testid="edit-reasoning"
          />
        </div>
      )}

      <div className="msg-editor-pane">
        {reasoning !== undefined && (
          <div className="msg-editor-label">
            <Icon name="chat" size={11} />
            <span>Reply</span>
          </div>
        )}
        <textarea
          ref={bodyRef}
          value={body}
          onChange={(e) => setBody(e.target.value)}
          onKeyDown={keys}
          data-testid="edit-content"
        />
      </div>

      <div className="msg-editor-actions">
        <span className="faint tiny-note">Ctrl+Enter to save · Esc to cancel</span>
        <button onClick={onCancel} data-testid="edit-cancel">
          Cancel
        </button>
        <button
          className="primary"
          onClick={save}
          disabled={saving || disabled}
          title={disabled ? disabledReason : undefined}
          data-testid="edit-save"
        >
          Save
        </button>
      </div>
    </div>
  )
}
