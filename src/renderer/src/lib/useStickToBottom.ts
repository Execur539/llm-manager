/**
 * Keeps a scroll container pinned to the bottom while content arrives, until the reader says not to.
 *
 * What this replaces was a `scrollIntoView({ behavior: 'smooth' })` on a sentinel element, fired
 * from a React effect whose dependencies listed the state the view happened to know about. It
 * came apart in three separate ways.
 *
 * The dependency list could not name everything that changes height. A tool result arrives by
 * mutating an entry already in the array, so the length never changes and the effect never ran —
 * the card grew by exactly the height of its output and the view stayed where it was. Streamed
 * reasoning was not in the list at all, so a thinking model scrolled nothing until its first
 * answer token. An image finishing loading, or markdown reflowing, changes height with no React
 * state involved whatsoever. Growth is a layout fact, so it is now observed as one.
 *
 * The scrolling was smooth, and `.messages` also sets `scroll-behavior: smooth` in CSS, so every
 * token restarted an animation from wherever the last one had reached. During a fast stream the
 * animation never finished before being superseded and the view trailed the true bottom
 * permanently — a tall element like a tool card opened a gap big enough to see. Pinning is
 * instant now, which cannot fall behind. Only the explicit jump is animated, because that one is
 * a deliberate movement the eye should be able to follow.
 *
 * And nothing tracked whether the reader wanted to be at the bottom. Scrolling up to re-read
 * something was fighting the next token.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * How close to the bottom still counts as following along.
 *
 * Not zero: sub-pixel rounding on a scaled display leaves fractional pixels of slack, and a
 * reader who nudges the wheel a notch has not asked to stop following. Roughly two lines.
 */
const NEAR_BOTTOM_PX = 72

export interface StickToBottom {
  /** Goes on the scrolling element. */
  scrollRef: React.RefObject<HTMLDivElement>
  /** Goes on a single wrapper around everything inside it — this is what is measured. */
  contentRef: React.RefObject<HTMLDivElement>
  /** True when the reader has scrolled away and new content is no longer being followed. */
  detached: boolean
  /** Return to the bottom and resume following. */
  jumpToLatest: () => void
}

export function useStickToBottom(resetKey: string | null): StickToBottom {
  const scrollRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  /*
   * A ref, not state, and read inside the observer.
   *
   * The observer fires many times a second during a stream. Reading state there would need it in
   * the dependency list, which would tear down and rebuild the observer on every change.
   */
  const following = useRef(true)
  const [detached, setDetached] = useState(false)

  const pin = useCallback((behavior: ScrollBehavior): void => {
    const el = scrollRef.current
    if (el) el.scrollTo({ top: el.scrollHeight, behavior })
  }, [])

  const jumpToLatest = useCallback((): void => {
    following.current = true
    setDetached(false)
    pin('smooth')
  }, [pin])

  /*
   * Whether we are following is derived from where the container actually is.
   *
   * Deliberately not a flag we set when scrolling ourselves. A pin lands at the bottom, so the
   * scroll event it causes measures a distance of zero and confirms what we already believed —
   * there is no feedback loop to suppress and no "is this scroll mine?" bookkeeping to get wrong.
   * Scrolling up detaches, scrolling back re-attaches, and both work while a response streams.
   */
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const onScroll = (): void => {
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight
      const next = distance <= NEAR_BOTTOM_PX
      if (next === following.current) return
      following.current = next
      setDetached(!next)
    }
    el.addEventListener('scroll', onScroll, { passive: true })
    return () => el.removeEventListener('scroll', onScroll)
  }, [])

  /*
   * Scrolling up detaches immediately, before the position says so.
   *
   * Position alone loses a race. A wheel notch moves the container a little and fires a scroll
   * event; if the next token arrives before that scroll has carried past the threshold, the
   * observer below still believes we are following and pins straight back to the bottom. The
   * reader's scroll is undone as fast as they make it, and during a fast response the transcript
   * simply refuses to be scrolled up at all.
   *
   * Intent is not ambiguous, so it does not need measuring: a wheel turned upwards, or Page Up,
   * means stop following now. Coming back is still decided by position, because arriving at the
   * bottom is the unambiguous signal for the opposite.
   */
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const detach = (): void => {
      if (!following.current) return
      following.current = false
      setDetached(true)
    }
    const onWheel = (e: WheelEvent): void => {
      if (e.deltaY < 0) detach()
    }
    const onKey = (e: KeyboardEvent): void => {
      if (['ArrowUp', 'PageUp', 'Home'].includes(e.key)) detach()
    }
    el.addEventListener('wheel', onWheel, { passive: true })
    el.addEventListener('keydown', onKey)
    return () => {
      el.removeEventListener('wheel', onWheel)
      el.removeEventListener('keydown', onKey)
    }
  }, [])

  /*
   * Any change in content height re-pins, whatever caused it.
   *
   * This is the part that makes it robust: it does not care whether the growth came from a token,
   * a tool result filling in, an image decoding or a window resize reflowing the text. Coalesced
   * to one pin per frame, since a stream can resize the content far more often than it paints.
   */
  useEffect(() => {
    const content = contentRef.current
    if (!content) return
    let frame = 0
    const observer = new ResizeObserver(() => {
      if (!following.current) return
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => pin('instant'))
    })
    observer.observe(content)
    return () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
    }
  }, [pin])

  /*
   * A different conversation opens at the bottom, following again.
   *
   * After a frame, because the messages for it have not been laid out at the point this runs, so
   * the scroll height is still the previous conversation's.
   */
  useEffect(() => {
    following.current = true
    setDetached(false)
    const frame = requestAnimationFrame(() => pin('instant'))
    return () => cancelAnimationFrame(frame)
  }, [resetKey, pin])

  return { scrollRef, contentRef, detached, jumpToLatest }
}
