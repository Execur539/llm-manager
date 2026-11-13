/**
 * The app's mark: a stack of layers, which is what a model is and what this app arranges across
 * your cards. A small tile in the accent, lit from above, like an app icon rather than a swatch.
 *
 * It replaces a pair of sparkles that made the app look like a chat product's sibling.
 */
export default function BrandMark(): JSX.Element {
  return (
    <span className="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 24 24" width="15" height="15" fill="none" focusable="false">
        <path d="M12 3.6 20.2 8 12 12.4 3.8 8 12 3.6Z" fill="currentColor" />
        <path d="m3.8 12 8.2 4.4 8.2-4.4" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" />
        <path
          d="m3.8 16 8.2 4.4 8.2-4.4"
          stroke="currentColor"
          strokeWidth="1.9"
          strokeLinecap="round"
          strokeLinejoin="round"
          opacity="0.55"
        />
      </svg>
    </span>
  )
}
