/**
 * A tool call the model is still writing out, shown inline where the finished card will appear.
 *
 * A call cannot be dispatched until all of its arguments have arrived, so between the model
 * deciding to act and the call being ready there was nothing on screen — no text, since it has
 * stopped writing prose, and no card, since the call does not exist yet. On a long argument like
 * a file's contents that is several seconds of an agent that appears to have stopped. This fills
 * that gap with the one thing worth knowing: which tool, and what it is pointed at.
 *
 * Replaced by the real card the moment the call completes.
 */

/**
 * Arguments worth naming, in the order they are worth naming them.
 *
 * The same keys the finished card summarises by, so a call does not appear to change what it is
 * about the instant it finishes.
 */
const PREVIEW_KEYS = ['path', 'file_path', 'command', 'query', 'url', 'task', 'pattern', 'job_id']

/**
 * Pull a readable target out of half-written JSON.
 *
 * The argument string is incomplete by definition — `{"path":"src/ma` is exactly what this is
 * here to render — so it cannot be parsed. The pattern deliberately does not require the closing
 * quote, which is what lets a path appear a character at a time as the model types it.
 */
function preview(args: string): string | null {
  for (const key of PREVIEW_KEYS) {
    const match = args.match(new RegExp(`"${key}"\\s*:\\s*"((?:[^"\\\\]|\\\\.)*)`))
    // Unescape only what JSON escaped; a half-written escape at the very end is simply dropped.
    if (match?.[1]) return match[1].replace(/\\(.)/g, '$1')
  }
  return null
}

export default function PendingToolCall({ name, args }: { name: string; args: string }): JSX.Element {
  const target = preview(args)
  return (
    <div className="msg-aside">
      <div className="tool-card pending" data-testid="pending-tool-call">
        <div className="pending-row">
          <span className="badge warn">writing</span>
          {/* The name arrives before the arguments, so it is briefly the only thing known. */}
          <span className="tool-name">{name || 'deciding…'}</span>
          {target && <span className="truncate tool-detail">{target}</span>}
          <span className="pending-caret" aria-hidden="true" />
        </div>
      </div>
    </div>
  )
}
