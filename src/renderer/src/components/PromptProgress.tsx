/**
 * What the server is doing between pressing send and the first token appearing.
 *
 * Reading the prompt is real work — for a long conversation, a pasted document, or a model
 * that has just been loaded, it is often the majority of the wait. Until now it rendered as
 * three animated dots, which say only "something is happening" and look identical whether the
 * server is nine tenths done or has hung.
 *
 * The percentage comes from llama.cpp itself rather than being estimated here, so it reflects
 * tokens actually ingested. A prefix already in the KV cache counts as done, which is why a
 * follow-up in a long conversation can legitimately open at a high number instead of at zero.
 */

interface Props {
  percent: number
  processed: number
  total: number
  /** Tokens served from the KV cache rather than re-read. Zero on the first turn. */
  cached: number
}

export default function PromptProgress({ percent, processed, total, cached }: Props): JSX.Element {
  return (
    <div className="prompt-progress" data-testid="prompt-progress">
      <div className="row" style={{ justifyContent: 'space-between', fontSize: 12 }}>
        <span>Processing your prompt…</span>
        <span className="mono">{percent}%</span>
      </div>
      <div className="meter" style={{ marginTop: 5 }}>
        <span style={{ width: `${percent}%` }} />
      </div>
      <div className="faint" style={{ fontSize: 11, marginTop: 5 }}>
        {processed.toLocaleString()} of {total.toLocaleString()} tokens read
        {cached > 0 && ` · ${cached.toLocaleString()} reused from cache`}
      </div>
    </div>
  )
}
