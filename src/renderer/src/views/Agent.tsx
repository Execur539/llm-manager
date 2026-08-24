import { useEffect, useRef, useState } from 'react'
import type { AgentMessage, ToolCall, ToolResult } from '@shared/types'
import { invoke, fmtDuration } from '../lib/api'
import { select, setRunning, takePending, dropPending, useStream, clearFor, clearNotice } from '../lib/store'
import type { LoadedModel } from '../App'
import ConversationList, { type ChatSummary } from '../components/ConversationList'
import Markdown from '../components/Markdown'

/** Collapsed by default; one line of summary, expanding to arguments and full output. */
function ToolCard({ call, result }: { call: ToolCall; result?: ToolResult }): JSX.Element {
  const a = call.args as Record<string, unknown>
  const detail = ['path', 'command', 'query', 'url', 'task', 'job_id', 'pattern']
    .map((key) => (typeof a[key] === 'string' ? (a[key] as string) : null))
    .find((v): v is string => !!v)

  return (
    <details className="tool-card" data-testid="tool-card">
      <summary>
        <span className={`badge ${result ? (result.ok ? 'good' : 'bad') : 'warn'}`}>
          {result ? (result.ok ? 'ok' : 'failed') : 'running'}
        </span>
        <span className="tool-name">{call.name}</span>
        {detail && <span className="truncate tool-detail">{detail}</span>}
        <span className="tool-meta">
          {result && <span className="faint">{fmtDuration(result.durationMs)}</span>}
          {result?.truncated && <span className="badge warn">truncated</span>}
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
  )
}

export default function AgentView({ loaded }: { loaded: LoadedModel | null }): JSX.Element {
  const [sessions, setSessions] = useState<ChatSummary[]>([])
  const [messages, setMessages] = useState<AgentMessage[]>([])
  const [input, setInput] = useState('')
  const [tools, setTools] = useState<{ name: string; tier: string; description: string }[]>([])
  const [cwd, setCwd] = useState('')
  const [planMode, setPlanMode] = useState(false)
  const [showTools, setShowTools] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)

  const stream = useStream()
  // Selection lives in the store so a remount restores the open session, not a blank pane.
  const activeId = stream.selection.agent
  const partial = activeId ? (stream.partial[activeId] ?? '') : ''
  const running = activeId ? !!stream.running[activeId] : false
  const error = activeId ? stream.errors[activeId] : null
  const notice = activeId ? stream.notices[activeId] : null
  const liveToolCalls = activeId ? (stream.toolCalls[activeId] ?? []) : []

  const refreshSessions = async (): Promise<void> => {
    setSessions(await invoke<ChatSummary[]>('chat:list', 'agent'))
  }

  useEffect(() => {
    void refreshSessions()
    void invoke<typeof tools>('agent:tools').then(setTools).catch(() => undefined)
    void invoke<{ agent: { planMode: boolean } }>('settings:get').then((s) => setPlanMode(s.agent.planMode))
  }, [])

  // Reload history when the selected session changes, including on remount.
  useEffect(() => {
    if (!activeId) {
      setMessages([])
      setCwd('')
      return
    }
    let cancelled = false
    void (async () => {
      const session = await invoke<{ messages: AgentMessage[]; cwd: string } | null>('chat:load', activeId)
      if (cancelled) return
      setMessages(session?.messages ?? [])
      setCwd(session?.cwd ?? '')
      dropPending(activeId)
    })()
    return () => {
      cancelled = true
    }
  }, [activeId])

  // Absorb anything that streamed in while this view was unmounted.
  useEffect(() => {
    if (!activeId) return
    const pending = takePending(activeId)
    if (pending.length) {
      setMessages((prev) => {
        const seen = new Set(prev.map((m) => m.id))
        return [...prev, ...pending.filter((m) => !seen.has(m.id))]
      })
      void refreshSessions()
    }
  }, [activeId, stream.pending])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, partial, liveToolCalls.length])

  const openSession = (id: string): void => {
    select('agent', id)
  }

  const newSession = async (): Promise<void> => {
    const s = await invoke<ChatSummary>('chat:create', { kind: 'agent', title: 'New session' })
    await refreshSessions()
    setMessages([])
    select('agent', s.id)
  }

  const deleteSession = async (id: string): Promise<void> => {
    await invoke('chat:delete', id)
    clearFor(id)
    await refreshSessions()
  }

  const send = async (): Promise<void> => {
    if (!input.trim() || running || !loaded) return

    let sessionId = activeId
    if (!sessionId) {
      const s = await invoke<ChatSummary>('chat:create', { kind: 'agent', title: 'New session' })
      sessionId = s.id
      setMessages([])
      select('agent', sessionId)
    }

    const text = input
    setInput('')
    setRunning(sessionId, true)

    try {
      await invoke('agent:run', sessionId, text)
    } finally {
      setRunning(sessionId, false)
      await refreshSessions()
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

  // Tool calls already saved as messages, plus any still streaming this turn.
  const persistedCallIds = new Set(
    messages.flatMap((m) => (m.toolCalls ?? []).map((c) => c.id))
  )
  const unsavedCalls = liveToolCalls.filter((entry) => !persistedCallIds.has(entry.call.id))

  return (
    <div className="split">
      <ConversationList
        items={sessions}
        activeId={activeId}
        newLabel="New session"
        emptyLabel="No sessions yet."
        runningIds={stream.running}
        onNew={() => void newSession()}
        onOpen={(id) => openSession(id)}
        onDelete={(id) => void deleteSession(id)}
        onRename={(id, title) => void invoke('chat:rename', id, title).then(refreshSessions)}
      />

      <div className="chat">
        <div className="row head" style={{ flexWrap: 'wrap' }}>
          <h1 style={{ marginRight: 'auto' }}>Agent</h1>
          <button className="link" onClick={() => setShowTools((v) => !v)} data-testid="toggle-tools">
            {tools.length} tools
          </button>
          <button
            className={planMode ? 'primary' : ''}
            onClick={() => void togglePlanMode()}
            title="Restrict the agent to read-only tools until you approve a plan"
            data-testid="plan-mode"
          >
            Plan mode {planMode ? 'on' : 'off'}
          </button>
          <button onClick={() => void pickCwd()} title={cwd || 'Choose the folder the agent works in'}>
            {cwd ? `📁 ${cwd.split(/[\\/]/).filter(Boolean).pop()}` : 'Set folder'}
          </button>
          <button onClick={() => void invoke('agent:compact')} title="Summarise older turns to free context">
            Compact
          </button>
          {running && (
            <button className="danger" onClick={() => void invoke('agent:stop')} data-testid="agent-stop">
              Stop
            </button>
          )}
        </div>

        {showTools && (
          <div className="card">
            <div className="card-title">Available tools</div>
            <div className="tool-grid">
              {tools.map((t) => (
                <span
                  key={t.name}
                  className={`badge ${t.tier === 'read' ? 'good' : t.tier === 'write' ? 'warn' : 'bad'}`}
                  title={`${t.tier}: ${t.description}`}
                >
                  {t.name}
                </span>
              ))}
            </div>
            <div className="faint tiny-note">
              Green runs freely. Amber and red ask for approval before anything happens.
            </div>
          </div>
        )}

        {!loaded && (
          <div className="card note">
            No model is loaded. Load one from <strong>My models</strong> first.
          </div>
        )}
        {planMode && <div className="card note">Plan mode is on — the agent can only read until you turn it off.</div>}
        {notice && (
          <div className="card note dismissible">
            <span>{notice}</span>
            <button className="tiny" onClick={() => activeId && clearNotice(activeId)}>
              Dismiss
            </button>
          </div>
        )}
        {error && (
          <div className="card error-card">
            <span className="badge bad">error</span> {error}
          </div>
        )}

        <div className="messages" data-testid="agent-messages">
          {!messages.length && !partial && !unsavedCalls.length && (
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
                {m.role === 'assistant' ? (
                  <Markdown source={m.content} />
                ) : (
                  <div className="body">{m.content}</div>
                )}
              </div>
            )
          )}

          {unsavedCalls.map((entry) => (
            <ToolCard key={entry.call.id} call={entry.call} result={entry.result} />
          ))}

          {partial && (
            <div className="msg" data-testid="streaming-message">
              <div className="who">assistant</div>
              <div className="body streaming">
                <Markdown source={partial} />
                <span className="cursor" />
              </div>
            </div>
          )}
          {running && !partial && !unsavedCalls.length && (
            <div className="msg">
              <div className="who">assistant</div>
              <div className="body dim">Working…</div>
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
            data-testid="agent-input"
          />
          <button
            className="primary"
            onClick={() => void send()}
            disabled={!loaded || running}
            data-testid="agent-send"
          >
            {running ? 'Working…' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  )
}
