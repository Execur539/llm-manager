import { useEffect, useRef, useState } from 'react'
import type { AgentMessage, ToolCall, ToolResult } from '@shared/types'
import { invoke, on, fmtRelative, fmtDuration } from '../lib/api'
import type { LoadedModel } from '../App'

interface SessionSummary {
  id: string
  title: string
  cwd: string | null
  updatedAt: number
  messageCount: number
}

/** Collapsed by default; one line of summary, expanding to arguments and full output. */
function ToolCard({ call, result }: { call: ToolCall; result?: ToolResult }): JSX.Element {
  const a = call.args as Record<string, unknown>
  // Pick the single most informative argument for the collapsed one-line summary.
  const detail = ['path', 'command', 'query', 'url', 'task', 'job_id']
    .map((key) => (typeof a[key] === 'string' ? (a[key] as string) : null))
    .find((v): v is string => !!v)
  const summary = detail ? `${call.name}  ${detail.slice(0, 80)}` : call.name

  return (
    <details className="tool-card">
      <summary>
        <span className={`badge ${result ? (result.ok ? 'good' : 'bad') : 'warn'}`}>
          {result ? (result.ok ? 'ok' : 'failed') : 'running'}
        </span>
        <span className="truncate" style={{ flex: 1 }}>{summary}</span>
        {result && <span className="faint">{fmtDuration(result.durationMs)}</span>}
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
            {result.fullOutputPath && `\n\n[full output kept at ${result.fullOutputPath}]`}
          </>
        )}
      </div>
    </details>
  )
}

export default function AgentView({ loaded }: { loaded: LoadedModel | null }): JSX.Element {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<AgentMessage[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState('')
  const [running, setRunning] = useState(false)
  const [tools, setTools] = useState<{ name: string; tier: string }[]>([])
  const [cwd, setCwd] = useState('')
  const [planMode, setPlanMode] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [compacted, setCompacted] = useState<string | null>(null)
  const [showTools, setShowTools] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)

  const refreshSessions = async (): Promise<void> => {
    setSessions(await invoke<SessionSummary[]>('chat:list', 'agent'))
  }

  useEffect(() => {
    void refreshSessions()
    void invoke<{ name: string; tier: string }[]>('agent:tools').then(setTools).catch(() => undefined)
    void invoke<{ agent: { planMode: boolean } }>('settings:get').then((s) => setPlanMode(s.agent.planMode))

    const offs = [
      on<string>('agent:delta', (t) => setStreaming((s) => s + t)),
      on<AgentMessage>('agent:message', (m) => {
        setStreaming('')
        setMessages((prev) => (prev.some((x) => x.id === m.id) ? prev : [...prev, m]))
      }),
      on<{ strategy: string }>('agent:compacted', (info) =>
        setCompacted(`Context compacted (${info.strategy}) to keep the session going.`)
      ),
      on<string>('agent:done', () => {
        setRunning(false)
        void refreshSessions()
      }),
      on<string>('agent:error', (e) => {
        setError(e)
        setRunning(false)
      })
    ]
    return () => offs.forEach((off) => off())
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streaming])

  const openSession = async (id: string): Promise<void> => {
    setActiveId(id)
    setError(null)
    const session = await invoke<{ messages: AgentMessage[]; cwd: string } | null>('chat:load', id)
    setMessages(session?.messages ?? [])
    setCwd(session?.cwd ?? '')
  }

  const newSession = async (): Promise<void> => {
    const s = await invoke<SessionSummary>('chat:create', { kind: 'agent', title: 'New session' })
    await refreshSessions()
    setActiveId(s.id)
    setMessages([])
  }

  const send = async (): Promise<void> => {
    if (!input.trim() || running || !loaded) return
    let sessionId = activeId
    if (!sessionId) {
      const s = await invoke<SessionSummary>('chat:create', { kind: 'agent', title: 'New session' })
      sessionId = s.id
      setActiveId(sessionId)
    }

    const text = input
    setInput('')
    setRunning(true)
    setError(null)
    setCompacted(null)
    try {
      await invoke('agent:run', sessionId, text)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setRunning(false)
    }
  }

  const togglePlanMode = async (): Promise<void> => {
    const next = !planMode
    setPlanMode(next)
    const s = await invoke<{ agent: Record<string, unknown> }>('settings:get')
    await invoke('settings:patch', { agent: { ...s.agent, planMode: next } })
  }

  const pickCwd = async (): Promise<void> => {
    const dir = await invoke<string | null>('agent:set-cwd', activeId ?? undefined)
    if (dir) setCwd(dir)
  }

  return (
    <div className="split">
      <aside className="side-list">
        <button className="primary" style={{ width: '100%', marginBottom: 10 }} onClick={() => void newSession()}>
          New session
        </button>
        {sessions.map((s) => (
          <div
            key={s.id}
            className={`side-item ${activeId === s.id ? 'active' : ''}`}
            onClick={() => void openSession(s.id)}
          >
            <div className="truncate">{s.title}</div>
            <div className="faint" style={{ fontSize: 10 }}>
              {s.messageCount} steps · {fmtRelative(s.updatedAt)}
            </div>
          </div>
        ))}
        {!sessions.length && <div className="empty" style={{ fontSize: 12 }}>No sessions yet.</div>}
      </aside>

      <div className="chat">
        <div className="row" style={{ marginBottom: 10, flexWrap: 'wrap' }}>
          <h1 style={{ marginRight: 'auto' }}>Agent</h1>
          <button className="link" onClick={() => setShowTools((v) => !v)}>
            {tools.length} tools
          </button>
          <button className={planMode ? 'primary' : ''} onClick={() => void togglePlanMode()} title="Read-only until you approve a plan">
            Plan mode {planMode ? 'on' : 'off'}
          </button>
          <button onClick={() => void pickCwd()} title={cwd || 'Set working directory'}>
            {cwd ? `📁 ${cwd.split(/[\\/]/).pop()}` : 'Set folder'}
          </button>
          <button onClick={() => void invoke('agent:compact')}>Compact</button>
          {running && (
            <button className="danger" onClick={() => void invoke('agent:stop')}>Stop</button>
          )}
        </div>

        {showTools && (
          <div className="card">
            <div className="card-title">Available tools</div>
            <div className="tool-grid">
              {tools.map((t) => (
                <span key={t.name} className={`badge ${t.tier === 'read' ? 'good' : t.tier === 'write' ? 'warn' : 'bad'}`}>
                  {t.name}
                </span>
              ))}
            </div>
            <div className="faint" style={{ fontSize: 11, marginTop: 8 }}>
              Green runs freely. Amber and red ask for approval before anything happens.
            </div>
          </div>
        )}

        {!loaded && <div className="card note">No model is loaded. Load one from <strong>My models</strong> first.</div>}
        {planMode && <div className="card note">Plan mode is on — the agent can only read until you turn it off.</div>}
        {compacted && <div className="card note">{compacted}</div>}
        {error && (
          <div className="card" style={{ borderColor: '#5c2626' }}>
            <span className="badge bad">error</span> {error}
          </div>
        )}

        <div className="messages">
          {!messages.length && !streaming && (
            <div className="empty">
              The agent can read and write files, run commands, browse the web, control the desktop and
              execute code.
              <br />
              Reads run freely; writes and commands ask first.
            </div>
          )}

          {messages.map((m) =>
            m.role === 'tool' && m.toolCalls?.[0] ? (
              <ToolCard key={m.id} call={m.toolCalls[0]} result={m.toolResult} />
            ) : (
              <div className="msg" key={m.id}>
                <div className="who">{m.role}</div>
                <div className="body">{m.content}</div>
              </div>
            )
          )}

          {streaming && (
            <div className="msg">
              <div className="who">assistant</div>
              <div className="body">{streaming}<span className="cursor" /></div>
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
            placeholder={loaded ? 'Ask the agent to do something…' : 'Load a model first'}
            disabled={!loaded || running}
          />
          <button className="primary" onClick={() => void send()} disabled={!loaded || running}>
            {running ? 'Working…' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  )
}
