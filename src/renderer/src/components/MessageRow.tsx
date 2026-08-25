/**
 * One turn in a transcript.
 *
 * Shared by Chat and Agent so the two transcripts cannot drift apart visually. The avatar column
 * is what makes the roles line up: previously the user's text was indented by its own bubble
 * padding while the assistant's started at the container edge, so the two were a few pixels out
 * of alignment down the whole conversation.
 */

import Icon from './Icon'

const LABELS: Record<string, string> = {
  user: 'You',
  assistant: 'Assistant',
  system: 'System',
  tool: 'Tool'
}

export default function MessageRow({
  role,
  children,
  testId
}: {
  role: string
  children: React.ReactNode
  testId?: string
}): JSX.Element {
  const label = LABELS[role] ?? role

  return (
    <div className={`msg from-${role}`} data-testid={testId}>
      <div className="msg-avatar" aria-hidden="true">
        <Icon name={role === 'user' ? 'user' : role === 'tool' ? 'chip' : 'sparkle'} size={13} />
      </div>
      <div className="msg-main">
        <div className="who">{label}</div>
        {children}
      </div>
    </div>
  )
}
