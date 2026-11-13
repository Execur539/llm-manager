/**
 * The window's shared chrome, filled in by whichever view is showing.
 *
 * The app draws its own title bar. Each view used to draw its own heading and row of buttons
 * inside the page, which scrolled away, sat at a different height on every view, and left the
 * window's top edge with nothing to drag. Now the toolbar belongs to the window and a view only
 * says what goes in it: a title and its actions (`Toolbar`), and, for the conversation views, the
 * list of conversations that lives in the sidebar on a wide window (`useSidebarSlot`).
 *
 * The slots are elements the shell renders and hands down once they exist, so the first render of
 * a view puts nothing there and the second does. Portals keep each view's own state and handlers
 * exactly where they were.
 */

import { createContext, useContext, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

export interface ShellSlots {
  /** Where a view's title goes, in the toolbar. */
  title: HTMLElement | null
  /** Where a view's actions go, at the toolbar's right, before the window's own buttons. */
  actions: HTMLElement | null
  /** The sidebar's space for a view's list (conversations, sessions), below the navigation. */
  sidebar: HTMLElement | null
  /** Whether the sidebar is showing icons only, in which case nothing can live in it. */
  collapsed: boolean
}

export const ShellContext = createContext<ShellSlots>({ title: null, actions: null, sidebar: null, collapsed: false })

export function useShell(): ShellSlots {
  return useContext(ShellContext)
}

/**
 * A view's title and actions, in the window toolbar.
 *
 * The title is the view's heading for assistive technology too, so it is an h1 wherever it lands.
 */
export function Toolbar({
  title,
  leading,
  children
}: {
  title?: ReactNode
  /** Before the title: the button that opens the conversations drawer on a narrow window. */
  leading?: ReactNode
  children?: ReactNode
}): JSX.Element {
  const slots = useShell()
  return (
    <>
      {slots.title && title !== undefined
        ? createPortal(
            <>
              {leading}
              <h1 className="titlebar-heading">{title}</h1>
            </>,
            slots.title
          )
        : null}
      {slots.actions && children ? createPortal(children, slots.actions) : null}
    </>
  )
}
