/**
 * One turn of a conversation.
 *
 * Your messages sit on the right in a bubble; the model's answers run as plain text across the
 * column, the way a reply reads in a document. Avatars and "You" / "Assistant" labels said the same
 * thing the layout already says, twice per message, so they are gone from sight and kept only for
 * screen readers. Other roles (a system note, a tool) keep a visible label, since nothing else
 * would tell them apart.
 */

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
  testId
}: {
  role: string
  children: React.ReactNode
  actions?: React.ReactNode
  testId?: string
}): JSX.Element {
  const label = LABELS[role] ?? role
  const quiet = role === 'user' || role === 'assistant'

  return (
    <div className={`msg from-${role}`} data-testid={testId}>
      {actions}
      <div className="msg-main">
        {quiet ? <span className="sr-only">{label}</span> : <div className="who">{label}</div>}
        {children}
      </div>
    </div>
  )
}
