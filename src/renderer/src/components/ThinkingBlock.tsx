/**
 * The model's chain of thought, in a block of its own.
 *
 * Reasoning models emit their thinking separately from the answer, and it is long, exploratory
 * and frequently wrong on the way to being right — so it is worth being able to read, and worth
 * being out of the way by default once the answer exists.
 *
 * While the model is still thinking the block is open and the text is live, because at that
 * point it is the only thing happening. As soon as the answer starts arriving it collapses,
 * unless the reader has opened it themselves — a panel that snaps shut under someone who is
 * reading it is worse than one that never opened.
 */

import { useEffect, useRef, useState } from 'react'
import Icon from './Icon'

export default function ThinkingBlock({
  text,
  streaming = false,
  answerStarted = false
}: {
  text: string
  /** The reasoning itself is still arriving. */
  streaming?: boolean
  /** The answer has begun, so the thinking is no longer the main event. */
  answerStarted?: boolean
}): JSX.Element | null {
  const [open, setOpen] = useState(streaming)
  // Once the reader has expressed a preference, stop overriding it.
  const touched = useRef(false)
  const bodyRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (touched.current) return
    setOpen(streaming && !answerStarted)
  }, [streaming, answerStarted])

  // Follow the thinking as it arrives, but only while it is the thing being watched.
  useEffect(() => {
    if (open && streaming) bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight })
  }, [text, open, streaming])

  if (!text.trim()) return null

  return (
    <div className={`thinking-block${open ? ' open' : ''}`} data-testid="thinking-block">
      <button
        className="thinking-head"
        onClick={() => {
          touched.current = true
          setOpen((v) => !v)
        }}
        aria-expanded={open}
        data-testid="thinking-toggle"
      >
        <Icon name="sparkle" size={12} />
        <span className="thinking-label">{streaming ? 'Thinking…' : 'Thought process'}</span>
        {!streaming && <span className="thinking-size">{approxWords(text)} words</span>}
        <Icon name="chevron" size={13} className="thinking-chevron" />
      </button>

      {open && (
        <div className="thinking-body" ref={bodyRef} data-testid="thinking-body">
          {text}
          {streaming && <span className="cursor" />}
        </div>
      )}
    </div>
  )
}

function approxWords(text: string): string {
  const n = text.trim().split(/\s+/).length
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n)
}
