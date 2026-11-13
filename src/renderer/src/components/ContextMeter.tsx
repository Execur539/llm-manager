/**
 * How much of the context window the conversation fills, as a small ring.
 *
 * A ring and a percentage say "how full" at a glance; the exact counts are in the tooltip for
 * anyone who wants them. It used to be a bar with "142/84.5K" beside it in monospace, which is a
 * sum to read rather than a level to see. It warms as the window fills, so the moment to compact
 * shows before the model starts forgetting.
 */

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n >= 10_000_000 ? 0 : 1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(n >= 100_000 ? 0 : 1)}K`
  return String(n)
}

export default function ContextMeter({ used, max }: { used: number; max: number }): JSX.Element | null {
  if (!max || max <= 0) return null

  const pct = Math.min(100, Math.max(0, (used / max) * 100))
  const tone = pct >= 90 ? ' full' : pct >= 75 ? ' filling' : ''
  const r = 6
  const length = 2 * Math.PI * r

  return (
    <div
      className={`context-meter${tone}`}
      title={`${used.toLocaleString()} of ${max.toLocaleString()} tokens of context used (${fmtTokens(used)} of ${fmtTokens(max)})`}
      data-testid="context-meter"
    >
      <svg className="context-ring" width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
        <circle cx="8" cy="8" r={r} className="context-ring-track" />
        <circle
          cx="8"
          cy="8"
          r={r}
          className="context-ring-fill"
          strokeDasharray={`${(pct / 100) * length} ${length}`}
          transform="rotate(-90 8 8)"
        />
      </svg>
      <span className="context-text">{pct < 1 && used > 0 ? '<1' : Math.round(pct)}%</span>
    </div>
  )
}
