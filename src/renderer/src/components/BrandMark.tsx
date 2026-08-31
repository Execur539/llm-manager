/**
 * The app's mark, top-left of the sidebar.
 *
 * It used to borrow the generic `sparkle` glyph from the icon set — the same four-point star that
 * labels an assistant message, a thinking block and an Ultra sample. Reusing it here meant the
 * one place that should say "this application" said exactly what four other things in the same
 * window were already saying, in the same shape.
 *
 * This is drawn for the tile instead: two stars rather than one, sized and placed against each
 * other so the mark still resolves at 26px, with concave sides that read as a sparkle rather
 * than a diamond. Left as flat `currentColor` on purpose — the tile behind it carries the
 * gradient and the depth, so the glyph only has to hold its silhouette.
 */
export default function BrandMark(): JSX.Element {
  return (
    <span className="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 24 24" width="17" height="17" fill="currentColor" focusable="false">
        {/* Primary star, pulled up and left to leave the companion room to sit clear of it. */}
        <path d="M10.4 1.7c.36 4.6 1.3 6.9 2.85 8.2 1.1.94 2.8 1.5 5.25 1.8-2.45.3-4.15.86-5.25 1.8-1.55 1.3-2.49 3.6-2.85 8.2-.36-4.6-1.3-6.9-2.85-8.2-1.1-.94-2.8-1.5-5.25-1.8 2.45-.3 4.15-.86 5.25-1.8 1.55-1.3 2.49-3.6 2.85-8.2Z" />
        {/* Companion, deliberately lighter so it reads as a highlight and not a second logo. */}
        <path
          d="M18.9 14.3c.2 2.3.72 3.35 1.7 3.85.42.22 1 .38 1.75.5-.75.12-1.33.28-1.75.5-.98.5-1.5 1.55-1.7 3.85-.2-2.3-.72-3.35-1.7-3.85-.42-.22-1-.38-1.75-.5.75-.12 1.33-.28 1.75-.5.98-.5 1.5-1.55 1.7-3.85Z"
          opacity="0.66"
        />
      </svg>
    </span>
  )
}
