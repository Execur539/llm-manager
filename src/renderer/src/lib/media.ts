/**
 * Subscribe to a CSS media query from React.
 *
 * Layout decisions belong in CSS, but a few behavioural ones follow the same breakpoint — a
 * closed drawer must not be focusable, and only the component knows whether it is a drawer at
 * this width. Reading the query keeps that single breakpoint in one place rather than hardcoding
 * a pixel value in a second one.
 */

import { useSyncExternalStore } from 'react'

export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const mql = window.matchMedia(query)
      mql.addEventListener('change', onChange)
      return () => mql.removeEventListener('change', onChange)
    },
    () => window.matchMedia(query).matches,
    // Server snapshot: never true, so nothing renders as a drawer before hydration.
    () => false
  )
}

/** The width at which the conversation rail becomes an overlay. Mirrors styles.css. */
export const DRAWER_QUERY = '(max-width: 900px)'
