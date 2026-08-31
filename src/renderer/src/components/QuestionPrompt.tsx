import { useEffect, useRef, useState } from 'react'
import { send } from '../lib/api'
import { dismissQuestion, useStream } from '../lib/store'

/**
 * The agent asking the user something, mid-turn.
 *
 * Deliberately not the permission dialog. That one asks whether an action may happen and has two
 * answers; this asks what the user wants and the answer is theirs to write. The turn is blocked
 * on it, which is the point — before this existed, an agent that hit a real ambiguity had to
 * guess and possibly do the wrong work, or stop and raise it in its final reply, losing the
 * context it had built up.
 *
 * The queue lives in the store, so a question raised while the user is on another page is still
 * waiting when they come back rather than being lost with the unmount.
 */
export default function QuestionPrompt(): JSX.Element | null {
  const { questionQueue } = useStream()
  const current = questionQueue[0]
  const [draft, setDraft] = useState('')
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // A fresh question starts from an empty box, and the box takes focus.
  useEffect(() => {
    if (!current) return
    setDraft('')
    const t = setTimeout(() => inputRef.current?.focus(), 30)
    return () => clearTimeout(t)
  }, [current?.id])

  if (!current) return null

  function answer(text: string): void {
    const value = text.trim()
    if (!value || !current) return
    send('agent:answer', current.id, value)
    dismissQuestion(current.id)
    setDraft('')
  }

  return (
    <div className="overlay" data-testid="question-overlay">
      <div className="modal" role="dialog" aria-modal="true" aria-label="The agent has a question">
        <h2>
          <span>A question</span>
          <span className="badge">agent</span>
        </h2>

        <p className="question-text" data-testid="question-text">
          {current.question}
        </p>

        {current.options.length > 0 && (
          <div className="question-options">
            {current.options.map((option) => (
              <button key={option} className="question-option" onClick={() => answer(option)} data-testid="question-option">
                {option}
              </button>
            ))}
          </div>
        )}

        <textarea
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            // Enter sends, as in the composer. Shift+Enter is a newline for a longer answer.
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              answer(draft)
            }
          }}
          placeholder={current.options.length ? 'Or answer in your own words…' : 'Your answer…'}
          rows={2}
          data-testid="question-input"
        />

        <div className="actions">
          {/*
            * No dismiss and no Escape.
            *
            * The agent is blocked until this is answered, and a dialog that can be waved away
            * would leave the turn hanging with nothing on screen to explain why. Stopping the
            * turn is the way out, and that settles the question from the main process.
            */}
          <span className="faint tiny-note">The agent is waiting. Stop the turn if you would rather not answer.</span>
          <button className="primary" onClick={() => answer(draft)} disabled={!draft.trim()} data-testid="question-send">
            Answer
          </button>
        </div>
      </div>
    </div>
  )
}
