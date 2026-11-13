import mark from '../assets/brand-mark.svg'

/**
 * The app's mark: a speech bubble cut into three layers, the same one the exe, the launcher and the
 * tray wear. Talking to a model is what the app is for; the slices are the model's layers in the
 * colours the load sheet gives a card's memory (weights, KV cache, working memory).
 *
 * Drawn by scripts/make-icon.mjs, which writes the SVG this shows, so the sidebar and the icon can
 * never drift apart. It replaced a stock "layers" glyph on a blue tile.
 */
export default function BrandMark(): JSX.Element {
  return <img className="brand-mark" src={mark} alt="" aria-hidden="true" width={22} height={22} draggable={false} />
}
