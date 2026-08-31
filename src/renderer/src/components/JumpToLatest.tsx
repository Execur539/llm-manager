import Icon from './Icon'

/**
 * The way back down, shown only once the reader has scrolled away from the bottom.
 *
 * Following stops as soon as you scroll up, which is the correct behaviour — a response should
 * not drag the page out from under someone re-reading an earlier part of it. But that leaves a
 * state where new content is arriving somewhere you cannot see, and the only way back is to
 * scroll the whole way by hand while it keeps growing underneath you.
 *
 * Floats over the transcript rather than sitting in the layout, so its appearance and departure
 * do not move a single line of the text being read.
 */
export default function JumpToLatest({
  show,
  onClick
}: {
  show: boolean
  onClick: () => void
}): JSX.Element {
  return (
    <button
      type="button"
      className={`jump-latest${show ? ' show' : ''}`}
      onClick={onClick}
      // Hidden from the tab order when it is not offering anything, rather than merely invisible.
      tabIndex={show ? 0 : -1}
      aria-hidden={show ? undefined : true}
      data-testid="jump-to-latest"
    >
      <Icon name="chevron" size={13} />
      Jump to latest
    </button>
  )
}
