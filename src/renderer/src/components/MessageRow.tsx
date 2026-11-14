/**
 * One turn of a conversation.
 *
 * Your messages sit on the right in a bubble; the model's answers run as plain text across the
 * column, the way a reply reads in a document. Avatars and "You" / "Assistant" labels said the same
 * thing the layout already says, twice per message, so they are gone from sight and kept only for
 * screen readers. Other roles (a system note, a tool) keep a visible label, since nothing else
 * would tell them apart.
 */

import { useRef } from 'react'
import { useSaveSplit, type SaveMark } from './SaveState'

const LABELS: Record<string, string> = {
  user: 'You',
  assistant: 'Assistant',
  system: 'System',
  tool: 'Tool'
}

export default function MessageRow({
  role,
  children,
  actions,
  testId,
  save
}: {
  role: string
  children: React.ReactNode
  actions?: React.ReactNode
  testId?: string
  /** Whether the conversation's saved model state holds this message; drawn as its edge. */
  save?: SaveMark | null
}): JSX.Element {
  const label = LABELS[role] ?? role
  const quiet = role === 'user' || role === 'assistant'
  const ref = useRef<HTMLDivElement>(null)
  useSaveSplit(ref, save)

  return (
    <div ref={ref} className={`msg from-${role}`} data-testid={testId} data-save={save?.state}>
      {actions}
      <div className="msg-main">
        {quiet ? <span className="sr-only">{label}</span> : <div className="who">{label}</div>}
        {children}
      </div>
    </div>
  )
}
