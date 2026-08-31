/**
 * How much of the model's context window this conversation is occupying.
 *
 * The numbers come from llama-server, not from an estimate: the prompt total is what the server
 * reports having read, and generated tokens are added as they stream. That distinction matters
 * because the obvious estimate — characters over four — is a rule of thumb for English prose and
 * badly wrong for code, punctuation-heavy text, or any non-Latin script, and it cannot see the
 * system prompt, the chat template's own markup, or attachments at all.
 *
 * Filling the window is not an error, so it does not shout. It warns at three quarters, where
 * there is still room to finish a thought, and turns red at nine tenths, where the next turn is
 * likely to trigger a compaction.
 */

/** Compact enough to sit in a toolbar: 4,096 reads as 4.1K, 131,072 as 131K. */
function fmtTokens(n: number): string {
  if (!Number.isFinite(n) || n < 0) return '—'
  if (n >= 100_000) return `${Math.round(n / 1000)}K`
  if (n >= 10_000) return `${(n / 1000).toFixed(1)}K`
  if (n >= 1_000) return `${(n / 1000).toFixed(1)}K`
  return String(n)
}

export default function ContextMeter({
  used,
  max
}: {
  used: number
  max: number
}): JSX.Element | null {
  // Nothing useful to say before a model is loaded or a turn has ever run.
  if (!max || max <= 0) return null

  const pct = Math.min(100, Math.max(0, (used / max) * 100))
  const tone = pct >= 90 ? ' full' : pct >= 75 ? ' filling' : ''

  return (
    <div
      className={`context-meter${tone}`}
      title={`${used.toLocaleString()} of ${max.toLocaleString()} tokens of context used (${Math.round(pct)}%)`}
      data-testid="context-meter"
    >
      <span className="context-bar" aria-hidden="true">
        <span style={{ width: `${pct}%` }} />
      </span>
      <span className="context-text mono">
        {fmtTokens(used)}/{fmtTokens(max)}
      </span>
    </div>
  )
}
