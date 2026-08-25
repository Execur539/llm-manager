/**
 * Busy indicators.
 *
 * `Spinner` is for a control that is working; `Skeleton` stands in for content that is on its
 * way. The distinction matters: a spinner where a list will appear tells you nothing about what
 * is coming, and a skeleton on a button is nonsense.
 */

export function Spinner({ size = 14 }: { size?: number }): JSX.Element {
  return (
    <span
      className="spinner"
      style={{ width: size, height: size }}
      role="status"
      aria-label="Working"
      data-testid="spinner"
    />
  )
}

/**
 * Placeholder rows shaped like the content they precede.
 *
 * Sized to the real thing so the layout does not jump when it arrives — a skeleton that is the
 * wrong height just moves the flicker rather than removing it.
 */
export function Skeleton({
  rows = 3,
  height = 54,
  className = ''
}: {
  rows?: number
  height?: number
  className?: string
}): JSX.Element {
  return (
    <div className={`skeleton-list ${className}`.trim()} aria-hidden="true" data-testid="skeleton">
      {Array.from({ length: rows }, (_, i) => (
        <div
          key={i}
          className="skeleton-row"
          // A stagger reads as loading rather than as a broken repeating pattern.
          style={{ height, animationDelay: `${i * 90}ms` }}
        />
      ))}
    </div>
  )
}
