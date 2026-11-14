import type { ToolCall, ToolResult } from '@shared/types'
import { fmtDuration } from '../lib/api'
import { useState } from 'react'
import type { SaveMark } from './SaveState'
import Icon from './Icon'

/** One picture the model was shown; gone quietly if its file has since gone. */
function ToolImage({ src, label }: { src: string; label: string }): JSX.Element | null {
  const [failed, setFailed] = useState(false)
  if (failed) return null
  return (
    <figure className="tool-image">
      <img src={src} alt={`What the model was shown: ${label}`} loading="lazy" onError={() => setFailed(true)} />
    </figure>
  )
}

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
 *
 * `images` are pictures a tool showed the model whole (read_image, a screenshot, the clipboard),
 * shown at a size worth looking at, so the user sees what the model is looking at as it does.
 */
export default function ToolCard({
  call,
  result,
  frames,
  images,
  save
}: {
  call: ToolCall
  result?: ToolResult
  frames?: string[]
  images?: string[]
  /** Whether the conversation's saved model state holds this call; drawn as its edge. */
  save?: SaveMark | null
}): JSX.Element {
  const detail = detailOf(call)
  return (
    <div className="msg-aside" data-save={save?.state}>
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
      {images && images.length > 0 && (
        <div className="tool-images" data-testid="tool-images">
          {images.map((src, i) => (
            <ToolImage key={src} src={src} label={`${detail ?? call.name}${images.length > 1 ? ` (${i + 1})` : ''}`} />
          ))}
        </div>
      )}
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
