import type { ToolCall, ToolResult } from '@shared/types'
import { fmtDuration } from '../lib/api'
import Icon from './Icon'

/** Seconds as m:ss, the way watch_video labels its frames. */
function clock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

/** The one argument that says what a call was about, for its summary line. */
function detailOf(call: ToolCall): string | null {
  const a = call.args as Record<string, unknown>
  const main = ['path', 'name', 'command', 'query', 'url', 'task', 'job_id', 'pattern']
    .map((key) => (typeof a[key] === 'string' ? (a[key] as string) : null))
    .find((v): v is string => !!v)
  if (call.name !== 'watch_video') return main ?? null
  // A look at part of a video says which part.
  const start = Number(a.start)
  const end = Number(a.end)
  const range =
    Number.isFinite(start) || Number.isFinite(end)
      ? ` · ${Number.isFinite(start) ? clock(start) : '0:00'}–${Number.isFinite(end) ? clock(end) : 'end'}`
      : ''
  return main ? `${main}${range}` : null
}

/**
 * A tool call in a transcript: what was called, on what, and how it went, opening onto the detail.
 *
 * Shared by the agent and by chat, whose only tools are its scratchpad's. `frames` are the pictures
 * a look at a video returned, shown under the line so the reader sees what the model saw; they are
 * held in memory with the scratchpad, so a reopened conversation shows the line without them.
 */
export default function ToolCard({
  call,
  result,
  frames
}: {
  call: ToolCall
  result?: ToolResult
  frames?: string[]
}): JSX.Element {
  const detail = detailOf(call)
  return (
    <div className="msg-aside">
      <details className="tool-card" data-testid="tool-card">
        <summary>
          {/* A dot carries the state; the word is there for screen readers and on hover. */}
          <span
            className={`tool-status ${result ? (result.ok ? 'ok' : 'failed') : 'running'}`}
            title={result ? (result.ok ? 'Finished' : 'Failed') : 'Running'}
          >
            <span className="sr-only">{result ? (result.ok ? 'ok' : 'failed') : 'running'}</span>
          </span>
          <span className="tool-name">{call.name}</span>
          {detail && <span className="truncate tool-detail">{detail}</span>}
          <span className="tool-meta">
            {result && <span className="faint">{fmtDuration(result.durationMs)}</span>}
            {result?.truncated && <span className="badge warn">truncated</span>}
            <Icon name="chevron" size={13} className="tool-chevron" />
          </span>
        </summary>
        <div className="detail">
          <div className="detail-label">Arguments</div>
          <pre className="detail-pre">{JSON.stringify(call.args, null, 2)}</pre>
          {result && (
            <>
              <div className="detail-label">Result</div>
              <pre className="detail-pre">{result.content}</pre>
              {result.fullOutputPath && <div className="faint tiny-note">Full output: {result.fullOutputPath}</div>}
            </>
          )}
        </div>
      </details>
      {frames && frames.length > 0 && (
        <div className="tool-frames" data-testid="tool-frames">
          {frames.map((src, i) => (
            <img key={i} src={src} alt={`Frame ${i + 1} the model looked at`} loading="lazy" />
          ))}
        </div>
      )}
    </div>
  )
}
