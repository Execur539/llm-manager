import { useEffect, useRef, useState } from 'react'
import type { AgentMessage, ToolCall, ToolResult } from '@shared/types'
import { invoke, fmtDuration } from '../lib/api'
import {
  select,
  setRunning,
  takePending,
  dropPending,
  useStream,
  clearFor,
  clearNotice,
  clearQuestions,
  setReasoning,
  adoptReasoning,
  DRAFT_AGENT
} from '../lib/store'
import type { LoadedModel } from '../App'
import ConversationList, { type ChatSummary } from '../components/ConversationList'
import Icon from '../components/Icon'
import Markdown from '../components/Markdown'
import MessageRow from '../components/MessageRow'
import ThinkingBlock from '../components/ThinkingBlock'
import UltraSamples from '../components/UltraSamples'
import RailToggle from '../components/RailToggle'
import ReasoningControl, { sendableChoice } from '../components/ReasoningControl'
import { AttachmentBar, DropZone, useAttachments } from '../components/Attachments'
import { Spinner } from '../components/Spinner'
import EmptyState from '../components/EmptyState'

/** Collapsed by default; one line of summary, expanding to arguments and full output. */
function ToolCard({ call, result }: { call: ToolCall; result?: ToolResult }): JSX.Element {
  const a = call.args as Record<string, unknown>
  const detail = ['path', 'command', 'query', 'url', 'task', 'job_id', 'pattern']
    .map((key) => (typeof a[key] === 'string' ? (a[key] as string) : null))
    .find((v): v is string => !!v)

  return (
    <div className="msg-aside">
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
    </div>
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
  const attachments = useAttachments()

  const stream = useStream()
  // Selection lives in the store so a remount restores the open session, not a blank pane.
  const activeId = stream.selection.agent
  const partial = activeId ? (stream.partial[activeId] ?? '') : ''
  const reasoning = activeId ? (stream.reasoningPartial[activeId] ?? '') : ''
  const running = activeId ? !!stream.running[activeId] : false
  const error = activeId ? stream.errors[activeId] : null
  const notice = activeId ? stream.notices[activeId] : null
  const liveToolCalls = activeId ? (stream.toolCalls[activeId] ?? []) : []
  // The agent view restores no session on mount, so this is null far more often than in chat —
  // which is what made the control look broken here first.
  const effortId = activeId ?? DRAFT_AGENT
  const ultra = activeId ? (stream.ultra[activeId] ?? []) : []
  const synthesising = activeId ? !!stream.ultraSynthesising[activeId] : false
  const ultraPlan = activeId ? stream.ultraPlan[activeId] : undefined

  /*
   * Ultra's attempt count, mirrored from settings.
   *
   * It is a setting rather than per-conversation state because it is a statement about this
   * machine's patience, not about one question. Written back on change so the choice survives a
   * restart, and read once on mount.
   */
  const [ultraSamples, setUltraSamplesState] = useState(3)

  useEffect(() => {
    void invoke<{ ultra?: { samples?: number } }>('settings:get')
      .then((s) => setUltraSamplesState(s.ultra?.samples ?? 3))
      .catch(() => undefined)
  }, [])

  const setUltraSamples = (next: number): void => {
    setUltraSamplesState(next)
    void invoke('settings:patch', { ultra: { samples: next } }).catch(() => undefined)
  }

  const refreshSessions = async (): Promise<void> => {
    setSessions(await invoke<ChatSummary[]>('chat:list', 'agent'))
  }

  useEffect(() => {
    void refreshSessions()
    void invoke<typeof tools>('agent:tools')
      .then(setTools)
      .catch(() => undefined)
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
      const session = await invoke<{
        messages: AgentMessage[]
        cwd: string
      } | null>('chat:load', activeId)
      if (cancelled) return
      setMessages(session?.messages ?? [])
      setCwd(session?.cwd ?? '')
      dropPending(activeId)
    })()
    return () => {
      cancelled = true
    }
  }, [activeId])

  /*
   * Absorb anything that streamed in while this view was unmounted.
   *
   * A message whose id is already on screen replaces it in place rather than being discarded.
   * The same id legitimately arrives twice: the user's turn is announced before Ultra starts
   * planning and stored again when the loop finally runs, and only the second copy carries the
   * chosen plan. Skipping known ids meant the transcript kept the earlier, plan-less copy — so
   * the plan vanished the moment the live box was cleared and only reappeared on reload.
   */
  useEffect(() => {
    if (!activeId) return
    const pending = takePending(activeId)
    if (pending.length) {
      setMessages((prev) => {
        const byId = new Map(prev.map((m) => [m.id, m]))
        for (const m of pending) byId.set(m.id, m)
        return [...byId.values()]
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
    const s = await invoke<ChatSummary>('chat:create', {
      kind: 'agent',
      title: 'New session'
    })
    await refreshSessions()
    setMessages([])
    adoptReasoning(DRAFT_AGENT, s.id)
    select('agent', s.id)
  }

  const deleteSession = async (id: string): Promise<void> => {
    await invoke('chat:delete', id)
    clearFor(id)
    await refreshSessions()
  }

  const send = async (): Promise<void> => {
    if ((!input.trim() && !attachments.items.length) || running || !loaded) return

    let sessionId = activeId
    if (!sessionId) {
      const s = await invoke<ChatSummary>('chat:create', {
        kind: 'agent',
        title: 'New session'
      })
      sessionId = s.id
      setMessages([])
      adoptReasoning(DRAFT_AGENT, sessionId)
      select('agent', sessionId)
    }

    // As in chat: `stream` predates the adopt, so the draft key is where the choice still is.
    // Narrowed to what this model can express — see sendableChoice.
    const effort = sendableChoice(loaded?.caps?.reasoning, stream.reasoning[effortId] ?? null)

    const text = input
    setInput('')
    setRunning(sessionId, true)

    try {
      const files = attachments.items.map((a) => a.path)
      attachments.clear()
      await invoke('agent:run', sessionId, text, effort, files)
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
  const persistedCallIds = new Set(messages.flatMap((m) => (m.toolCalls ?? []).map((c) => c.id)))
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

      <DropZone onFiles={(f) => void attachments.addFiles(f)} disabled={running}>
      <div className="chat">
        <div className="row head chat-head">
          <RailToggle />
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
            <button className="danger" onClick={() => {
                // The main process settles any outstanding question when the turn is drained;
                // clearing here takes the dialog off screen at the same moment.
                clearQuestions()
                void invoke('agent:stop')
              }} data-testid="agent-stop">
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
            <EmptyState
              icon="agent"
              title="Put the model to work"
              body="It can read and write files, run commands, browse the web, control the desktop and execute code."
              hint="Reads run freely. Writes and commands ask first."
            />
          )}

          {messages.map((m) =>
            m.role === 'tool' && m.toolCalls?.[0] ? (
              <ToolCard key={m.id} call={m.toolCalls[0]} result={m.toolResult} />
            ) : (
              <MessageRow role={m.role} key={m.id}>
                {m.role === 'assistant' ? (
                <>
                  {m.reasoning && <ThinkingBlock text={m.reasoning} />}
                  {/* A turn can be reasoning only — the model thought, then called a tool
                      without a word of prose. Rendering an empty body leaves a stray gap. */}
                  {m.content && <Markdown source={m.content} />}
                </>
              ) : (
                <>
                  <div className="body">{m.content}</div>
                  {/* Kept with the turn it directed, so reading the transcript back explains
                      why the agent did what it did. */}
                  {m.plan && (
                    <div className="ultra-block turn-plan">
                      <div className="ultra-head">
                        <Icon name="sparkle" size={12} />
                        <span>Ultra — plan for this turn</span>
                      </div>
                      <div className="ultra-plan">
                        <div className="ultra-pre">{m.plan}</div>
                      </div>
                    </div>
                  )}
                </>
              )}
              </MessageRow>
            )
          )}

          {unsavedCalls.map((entry) => (
            <ToolCard key={entry.call.id} call={entry.call} result={entry.result} />
          ))}

          {(partial || reasoning || ultra.length > 0) && (
            <MessageRow role="assistant" testId="streaming-message">
              {/* Plans, while they are being weighed — above whatever the run then does. */}
              <UltraSamples samples={ultra} synthesising={synthesising} plan={ultraPlan} />
              {reasoning && <ThinkingBlock text={reasoning} streaming answerStarted={!!partial} />}
              <div className="body streaming">
                <Markdown source={partial} caret />
              </div>
            </MessageRow>
          )}
          {running && !partial && !reasoning && !unsavedCalls.length && !ultra.length && (
            <MessageRow role="assistant">
              <div className="thinking">
                <span className="dot" />
                <span className="dot" />
                <span className="dot" />
              </div>
            </MessageRow>
          )}
          <div ref={endRef} />
        </div>

        <div className="composer">
          <AttachmentBar items={attachments.items} onRemove={attachments.remove} disabled={running} />
          <div className="composer-shell">
            <button
              className="attach-button"
              onClick={() => void attachments.pick()}
              disabled={running || attachments.busy}
              title="Attach images, video, audio, or text files"
              aria-label="Attach files"
              data-testid="attach-button"
            >
              {attachments.busy ? <Spinner size={14} /> : <Icon name="plus" size={15} />}
            </button>
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
              rows={1}
              data-testid="agent-input"
            />
            {/* Same treatment as chat: the button stops the turn it started. */}
            <button
              className={`send-button${running ? ' stopping' : ''}`}
              onClick={() => {
                if (!running) return void send()
                clearQuestions()
                void invoke('agent:stop')
              }}
              disabled={!loaded || (!running && !input.trim() && !attachments.items.length)}
              title={running ? 'Stop the agent' : 'Send  (Enter)'}
              aria-label={running ? 'Stop' : 'Send'}
              data-testid="agent-send"
            >
              <Icon name={running ? 'stop' : 'send'} size={15} />
            </button>
          </div>
          <div className="composer-meta">
            <div className="composer-hint">Enter to send · Shift+Enter for a newline</div>
            <ReasoningControl
              support={loaded?.caps?.reasoning}
              value={stream.reasoning[effortId] ?? null}
              onChange={(next) => setReasoning(effortId, next)}
              disabled={running}
              samples={ultraSamples}
              onSamplesChange={setUltraSamples}
            />
          </div>
        </div>
      </div>
      </DropZone>
    </div>
  )
}
