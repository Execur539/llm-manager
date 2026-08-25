/**
 * The "nothing here yet" state.
 *
 * Previously a line of dim text pinned to the top of an otherwise empty pane, which read as a
 * loading failure rather than an invitation. Centring it and giving it a clear hierarchy — mark,
 * heading, explanation — makes an empty view look deliberate.
 */

import Icon, { type IconName } from './Icon'

export default function EmptyState({
  icon,
  title,
  body,
  hint,
  action
}: {
  icon: IconName
  title: string
  body?: string
  /** A quieter second line, for caveats rather than explanation. */
  hint?: string
  action?: React.ReactNode
}): JSX.Element {
  return (
    <div className="empty-state" data-testid="empty-state">
      <div className="empty-mark" aria-hidden="true">
        <Icon name={icon} size={22} strokeWidth={1.5} />
      </div>
      <div className="empty-title">{title}</div>
      {body && <p className="empty-body">{body}</p>}
      {hint && <p className="empty-hint">{hint}</p>}
      {action && <div className="empty-action">{action}</div>}
    </div>
  )
}
