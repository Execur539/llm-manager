/**
 * Ultra's independent attempts, above the answer they produced.
 *
 * They are shown, rather than hidden behind a spinner, because a turn that takes four times as
 * long has to account for the wait — and because seeing where the attempts disagreed is the most
 * honest signal available about how much to trust what came out. They are also visibly drafts:
 * collapsed, quieter than the answer, and gone once the turn ends.
 */

import { useState } from 'react'
import type { UltraSampleView } from '../lib/store'
import Icon from './Icon'
import { Spinner } from './Spinner'

function Sample({ sample, running }: { sample: UltraSampleView; running: boolean }): JSX.Element {
  const [open, setOpen] = useState(false)
  const words = sample.answer.trim() ? sample.answer.trim().split(/\s+/).length : 0

  return (
    <div className={`ultra-sample${open ? ' open' : ''}`} data-testid="ultra-sample">
      <button className="ultra-sample-head" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span className="ultra-index">{sample.index + 1}</span>
        <span className="ultra-sample-label">
          {running ? 'Attempting…' : `Attempt ${sample.index + 1}`}
        </span>
        <span className="ultra-sample-meta">
          {running && <Spinner />}
          {!running && sample.continuations ? (
            /* Worth surfacing: it is the part of Ultra that is not just "run it again". */
            <span className="ultra-tag" title="Times the model was pushed to keep thinking">
              +{sample.continuations} forced
            </span>
          ) : null}
          {!running && words > 0 && <span className="faint">{words} words</span>}
        </span>
        <Icon name="chevron" size={13} className="ultra-chevron" />
      </button>

      {open && (
        <div className="ultra-sample-body">
          {sample.reasoning.trim() && (
            <>
              <div className="ultra-sub">Thinking</div>
              <div className="ultra-pre">{sample.reasoning}</div>
            </>
          )}
          <div className="ultra-sub">Answer</div>
          <div className="ultra-pre">{sample.answer.trim() || '(nothing yet)'}</div>
        </div>
      )}
    </div>
  )
}

export default function UltraSamples({
  samples,
  synthesising
}: {
  samples: UltraSampleView[]
  synthesising: boolean
}): JSX.Element | null {
  if (!samples.length) return null

  return (
    <div className="ultra-block" data-testid="ultra-samples">
      <div className="ultra-head">
        <Icon name="sparkle" size={12} />
        <span>
          Ultra — {samples.length} {samples.length === 1 ? 'attempt' : 'attempts'}
        </span>
        {synthesising && <span className="ultra-status">comparing…</span>}
      </div>

      {samples.map((s) => (
        <Sample key={s.index} sample={s} running={!s.done} />
      ))}
    </div>
  )
}
