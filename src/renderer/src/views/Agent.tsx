import { useEffect, useRef, useState } from 'react'
import type { AgentMessage, AgentSessionState, ToolCall, ToolResult } from '@shared/types'

/** Collapsed tool card: one-line summary, expands to full arguments and output. */
function ToolCard({ call, result }: { call: ToolCall; result?: ToolResult }): JSX.Element {
  const summary = (): string => {
    const a = call.args as Record<string, unknown>
    if (a.path) return `${call.name} ${String(a.path)}`
    if (a.command) return `${call.name} ${String(a.command).slice(0, 70)}`
    if (a.query) return `${call.name} "${String(a.query)}"`
    if (a.url) return `${call.name} ${String(a.url)}`
    return call.name
  }

  return (
    <details className="tool-card">
      <summary>
        <span className={`badge ${result ? (result.ok ? 'good' : 'bad') : 'warn'}`}>
          {result ? (result.ok ? 'ok' : 'failed') : 'running'}
        </span>
        <span>{summary()}</span>
        {result && <span className="faint">{result.durationMs}ms</span>}
        {result?.truncated && <span className="badge warn">truncated</span>}
      </summary>
      <div className="detail">
        <strong>Arguments</strong>
        {'\n'}
        {JSON.stringify(call.args, null, 2)}
        {result && (
          <>
            {'\n\n'}
            <strong>Result</strong>
            {'\n'}
            {result.content}
            {result.fullOutputPath && `\n\n[full output: ${result.fullOutputPath}]`}
          </>
        )}
      </div>
    </details>
  )
}

export default function AgentView({ modelLoaded }: { modelLoaded: boolean }): JSX.Element {
  const [session, setSession] = useState<AgentSessionState>({
    id: `s-${Date.now()}`,
    title: 'Session',
    cwd: '',
    planMode: false,
    messages: [],
    taskList: [],
    createdAt: Date.now(),
    updatedAt: Date.now()
  })
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState('')
  const [running, setRunning] = useState(false)
  const [tools, setTools] = useState<{ name: string; tier: string; description: string }[]>([])
  const [cwd, setCwd] = useState<string>('')
  const [error, setError] = useState<string | null>(null)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    void (async () => setTools(await window.api.agent.tools()))()

    window.api.agent.onDelta((t) => setStreaming((s) => s + t))
    window.api.agent.onMessage((m) => {
      setStreaming('')
      setSession((s) => ({ ...s, messages: [...s.messages, m as AgentMessage] }))
    })
    window.api.agent.onDone(() => setRunning(false))
    window.api.agent.onError((e) => {
      setError(e)
      setRunning(false)
    })
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [session.messages, streaming])

  const send = async (): Promise<void> => {
    if (!input.trim() || running) return
    const text = input
    setInput('')
    setRunning(true)
    setError(null)
    try {
      await window.api.agent.run(session, text)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setRunning(false)
    }
  }

  const pickCwd = async (): Promise<void> => {
    const dir = await window.api.agent.setCwd()
    if (dir) setCwd(dir as string)
  }

  return (
    <div className="chat">
      <div className="row" style={{ marginBottom: 10, flexWrap: 'wrap' }}>
        <h1 style={{ marginRight: 'auto' }}>Agent</h1>
        <span className="badge">{tools.length} tools</span>
        <button onClick={() => void pickCwd()}>{cwd || 'Set working directory'}</button>
        {running && (
          <button className="danger" onClick={() => void window.api.agent.stop()}>
            Stop
          </button>
        )}
      </div>

      {!modelLoaded && (
        <div className="card badge warn" style={{ display: 'block' }}>
          No model is loaded. Load one from the Models tab before running the agent.
        </div>
      )}

      {error && (
        <div className="card" style={{ borderColor: '#5c2626' }}>
          <span className="badge bad">error</span> {error}
        </div>
      )}

      <div className="messages">
        {session.messages.length === 0 && !streaming && (
          <div className="empty">
            The agent has filesystem, shell, web and code-execution tools.
            <br />
            Reads run freely; writes and commands ask first.
          </div>
        )}

        {session.messages.map((m) => (
          <div className="msg" key={m.id}>
            {m.role === 'tool' && m.toolCalls?.[0] ? (
              <ToolCard call={m.toolCalls[0]} result={m.toolResult} />
            ) : (
              <>
                <div className="who">{m.role}</div>
                <div className="body">{m.content}</div>
              </>
            )}
          </div>
        ))}

        {streaming && (
          <div className="msg">
            <div className="who">assistant</div>
            <div className="body">{streaming}</div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="composer">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              void send()
            }
          }}
          placeholder={modelLoaded ? 'Ask the agent to do something…' : 'Load a model first'}
          disabled={!modelLoaded || running}
        />
        <button className="primary" onClick={() => void send()} disabled={!modelLoaded || running}>
          {running ? 'Working…' : 'Send'}
        </button>
      </div>
    </div>
  )
}
