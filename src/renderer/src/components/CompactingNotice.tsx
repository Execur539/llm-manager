import { useEffect, useState } from 'react'
import { Spinner } from './Spinner'

/**
 * Shown while the session's history is being summarised down to fit the context window.
 *
 * Compaction is a whole model call over the older half of the session. On a long one that runs
 * for many seconds, and it used to happen in total silence between iterations of a turn — the
 * app simply stopped producing output, with nothing to say work was under way or when it would
 * end. Silence is indistinguishable from a hang, which is why the elapsed time is here: a number
 * that keeps moving is the difference between "this is taking a while" and "this has died".
 *
 * The clock ticks four times a second and displays whole seconds, so it never looks stalled
 * between updates while still only ever showing a figure that is true.
 *
 * Once the summary runs in passes it also says which pass, how far through the whole it is, and
 * which model is doing it — the percentage is what makes a long compaction something to wait
 * for rather than something to wonder about.
 */
export default function CompactingNotice({
  since,
  automatic,
  percent,
  chunk,
  chunks,
  stage,
  helper
}: {
  since: number
  automatic: boolean
  percent?: number
  chunk?: number
  chunks?: number
  stage?: 'chunk' | 'merge'
  helper?: string | null
}): JSX.Element {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const tick = (): void => setElapsed(Math.max(0, Math.round((Date.now() - since) / 1000)))
    tick()
    const timer = setInterval(tick, 250)
    return () => clearInterval(timer)
  }, [since])

  const step = chunks ? (stage === 'merge' ? 'merging the summaries' : `part ${chunk} of ${chunks}`) : null

  return (
    <div className="compacting-notice" role="status" data-testid="compacting">
      <Spinner size={13} />
      <span className="compacting-text">
        {automatic
          ? 'The context window filled up — summarising earlier turns to make room'
          : 'Summarising earlier turns'}
        {step && (
          <span className="faint">
            {' · '}
            {step}
            {helper ? ` with ${helper}` : ''}
          </span>
        )}
      </span>
      {percent !== undefined && (
        <span className="mono compacting-elapsed" data-testid="compacting-percent">
          {percent}%
        </span>
      )}
      <span className="mono compacting-elapsed">{elapsed}s</span>
    </div>
  )
}
