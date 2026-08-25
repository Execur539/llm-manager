/**
 * Opens the conversation rail on viewports too narrow to keep it permanently.
 *
 * The desktop window has a 1040px minimum, so this only ever appears in the remote browser UI
 * on a phone or a small tablet. CSS decides when it is visible; the component itself is always
 * rendered so the two never disagree about the breakpoint.
 */

import { toggleRail, useStream } from '../lib/store'
import Icon from './Icon'

export default function RailToggle(): JSX.Element {
  const { railOpen } = useStream()

  return (
    <button
      className="rail-toggle"
      onClick={toggleRail}
      aria-expanded={railOpen}
      aria-label={railOpen ? 'Hide conversations' : 'Show conversations'}
      title={railOpen ? 'Hide conversations' : 'Show conversations'}
      data-testid="rail-toggle"
    >
      <Icon name={railOpen ? 'close' : 'chat'} size={15} />
    </button>
  )
}
