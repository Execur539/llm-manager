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
  seedContext,
  toast,
  DRAFT_AGENT
} from '../lib/store'
import type { LoadedModel } from '../App'
import ConversationList, { type ChatSummary } from '../components/ConversationList'
import Icon from '../components/Icon'
import Markdown from '../components/Markdown'
import MessageRow from '../components/MessageRow'
import MessageActions from '../components/MessageActions'
import MessageEditor from '../components/MessageEditor'
import MessageMedia, { stripAttachmentLine } from '../components/MessageMedia'
import ThinkingBlock from '../components/ThinkingBlock'
import UltraSamples from '../components/UltraSamples'
import RailToggle from '../components/RailToggle'
import ReasoningControl, { sendableChoice } from '../components/ReasoningControl'
import { AttachmentBar, DropZone, useAttachments } from '../components/Attachments'
import { Spinner } from '../components/Spinner'
import PromptProgress from '../components/PromptProgress'
import MediaStage from '../components/MediaStage'
import ContextMeter from '../components/ContextMeter'
import CompactingNotice from '../components/CompactingNotice'
import PendingToolCall from '../components/PendingToolCall'
import JumpToLatest from '../components/JumpToLatest'
import { useStickToBottom } from '../lib/useStickToBottom'
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
  /** The message currently open for rewriting, at most one at a time. */
  const [editing, setEditing] = useState<string | null>(null)
  const attachments = useAttachments()

  const stream = useStream()
  // Selection lives in the store so a remount restores the open session, not a blank pane.
  const activeId = stream.selection.agent
  const partial = activeId ? (stream.partial[activeId] ?? '') : ''
  const reasoning = activeId ? (stream.reasoningPartial[activeId] ?? '') : ''
  const running = activeId ? !!stream.running[activeId] : false
  const progress = activeId ? (stream.promptProgress[activeId] ?? null) : null
  const media = activeId ? (stream.mediaStage[activeId] ?? null) : null
  const ctx = activeId ? (stream.context[activeId] ?? null) : null
  const compacting = activeId ? (stream.compacting[activeId] ?? null) : null
  /*
   * Sending is blocked while the history is being rewritten.
   *
   * A message accepted mid-compaction is appended to a history that is about to be replaced by a
   * summary of itself, so it is either summarised away before the model ever reads it or lands
   * after turns it was meant to follow. Treated exactly like a turn in flight, because from the
   * composer's point of view that is what it is: the session is busy and not accepting input.
   */
  const busy = running || !!compacting

  // Keyed on the session, so switching to another one opens at its latest message.
  const { scrollRef, contentRef, detached, jumpToLatest } = useStickToBottom(activeId)
  const error = activeId ? stream.errors[activeId] : null
  const notice = activeId ? stream.notices[activeId] : null
  const liveToolCalls = activeId ? (stream.toolCalls[activeId] ?? []) : []
  const pendingCalls = activeId ? (stream.pendingCalls[activeId] ?? []) : []
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
        contextUsed?: number
      } | null>('chat:load', activeId)
      if (cancelled) return
      setMessages(session?.messages ?? [])
      setCwd(session?.cwd ?? '')
      // An id from the previous conversation would otherwise match nothing and edit nothing.
      setEditing(null)
      // So the meter reads correctly on reopening rather than staying blank until the next turn.
      seedContext(activeId, session?.contextUsed, loaded?.plan.contextLength ?? 0)
      dropPending(activeId)
    })()
    return () => {
      cancelled = true
    }
    // Re-seeded when a model loads, since the window's size comes from the model, not the session.
  }, [activeId, loaded?.plan.contextLength])

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
    if ((!input.trim() && !attachments.items.length) || busy || !loaded) return

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

  /**
   * Rewrite a turn that has already been said.
   *
   * An assistant turn is corrected in place: the stored message is replaced and the agent's
   * rolling history is dropped, so the next turn is built from the edited transcript rather than
   * from the words the model actually produced. Nothing is re-run — an edit changes what
   * happened, it does not ask for it to happen again.
   *
   * Editing your own turn means something different: the reply that followed was an answer to
   * words that no longer exist, so it goes with it, and the model answers the new wording fresh.
   * That is a full turn, not an instant round trip, so it goes through the same running state as
   * sending a message — the composer disables, the stop button appears, and the whole updated
   * tail of the conversation (not just the one edited bubble) replaces what is on screen.
   */
  const saveEdit = async (id: string, content: string, role: string, reasoning?: string): Promise<void> => {
    if (!activeId) return
    /*
     * Not while the model is busy, and not with no model to answer.
     *
     * The edit button that opens this editor is already hidden while busy, but the editor itself
     * has no such guard once open — start a turn elsewhere while it is sitting there, or open it
     * when nothing is loaded, and Save is still clickable. Checked here rather than trusting the
     * caller: this is the one place that actually knows whether it is about to run a turn.
     */
    if (busy) {
      toast('The model is busy — wait for it to finish first.', 'info')
      return
    }
    if (role === 'user' && !loaded) {
      toast('Load a model before rewinding a message.', 'info')
      return
    }
    setEditing(null)
    if (role === 'user') {
      /*
       * Rewound on screen before the network round trip, not after.
       *
       * The new reply streams in live through the same pending-message plumbing any other turn
       * uses — but nothing removes a message from that plumbing, so the stale reply and tool
       * calls this edit is about to erase would sit on screen next to the new one arriving until
       * the whole turn finished. Cutting the tail locally the moment the edit is confirmed is
       * what makes it look like what it is: the conversation being rewound, then answered again.
       */
      setMessages((prev) => {
        const idx = prev.findIndex((m) => m.id === id)
        if (idx === -1) return prev
        return [...prev.slice(0, idx), { ...prev[idx], content }]
      })

      const effort = sendableChoice(loaded?.caps?.reasoning, stream.reasoning[effortId] ?? null)
      setRunning(activeId, true)
      try {
        const session = await invoke<{ messages: AgentMessage[] }>(
          'agent:rewind-and-regenerate',
          activeId,
          id,
          content,
          effort
        )
        // The authoritative final shape, reconciling anything the streamed events did not carry.
        setMessages(session.messages)
      } catch (err) {
        toast(err instanceof Error ? err.message : String(err), 'error')
        /*
         * The cut above already happened on screen before the server ever saw the request, so a
         * rejection — no model loaded, another turn using the agent, the session having vanished
         * — leaves the transcript showing fewer messages than storage actually has. Reloading is
         * what `deleteMessage` does for the same reason: the server is the only source of truth
         * once the optimistic guess and reality can have parted ways.
         */
        const restored = await invoke<{ messages: AgentMessage[] } | null>('chat:load', activeId)
        if (restored) setMessages(restored.messages)
      } finally {
        setRunning(activeId, false)
        await refreshSessions()
      }
      return
    }
    try {
      const edited = await invoke<AgentMessage>('agent:edit-message', activeId, id, content, reasoning)
      setMessages((prev) => prev.map((m) => (m.id === id ? edited : m)))
    } catch (err) {
      toast(err instanceof Error ? err.message : String(err), 'error')
    }
  }

  /**
   * Remove a message, and everything the transcript shows after it.
   *
   * The row is gone from the screen the moment the confirm click lands — there is nothing to
   * stream back and wait for here, unlike an edit that regenerates. If the message being open
   * for editing is one of the ones removed, that editor has nothing left to edit.
   */
  const deleteMessage = async (id: string): Promise<void> => {
    if (!activeId) return
    const idx = messages.findIndex((m) => m.id === id)
    if (idx === -1) return
    setMessages((prev) => prev.slice(0, idx))
    if (editing && messages.slice(idx).some((m) => m.id === editing)) setEditing(null)
    try {
      await invoke('agent:delete-message', activeId, id)
    } catch (err) {
      toast(err instanceof Error ? err.message : String(err), 'error')
      // The optimistic cut was wrong if the server refused it — put the transcript back.
      const session = await invoke<{ messages: AgentMessage[] } | null>('chat:load', activeId)
      if (session) setMessages(session.messages)
    } finally {
      await refreshSessions()
    }
  }

  /**
   * Pick the last answer back up where it stopped.
   *
   * Offered only when the model's own turn is the last thing in the transcript — with a message
   * of yours after it there is nothing to continue, and the composer is what you want.
   */
  const continueAnswer = async (): Promise<void> => {
    if (!activeId || busy || !loaded) return
    const effort = sendableChoice(loaded?.caps?.reasoning, stream.reasoning[effortId] ?? null)
    setRunning(activeId, true)
    try {
      const session = await invoke<{ messages: AgentMessage[] }>('agent:continue', activeId, effort)
      setMessages(session.messages)
    } catch (err) {
      toast(err instanceof Error ? err.message : String(err), 'error')
    } finally {
      setRunning(activeId, false)
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

  /*
   * There is an answer to continue when the transcript ends on something the model produced.
   *
   * A tool result counts: the model was cut off between deciding what to do and saying anything
   * about it, which is exactly the case where carrying on is worth offering. A message of your
   * own after it does not — that is a question waiting to be sent, not an answer waiting to be
   * finished.
   */
  const canContinue =
    !!activeId && !!loaded && !busy && !editing && messages.length > 0 && messages.at(-1)?.role !== 'user'

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

      <DropZone onFiles={(f) => void attachments.addFiles(f)} disabled={busy}>
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
          {/*
            * Names the session, and says what happened.
            *
            * It used to send no argument and report nothing. With nothing to identify the
            * conversation the main process compacted whatever was last in memory — often
            * nothing at all — and either way the button looked identical: no message, no change
            * on screen, and no way to tell success from a silent no-op.
            */}
          <button
            disabled={!activeId || busy}
            onClick={async () => {
              if (!activeId) return
              const r = await invoke<{ ok: boolean; message: string; beforeTokens?: number; afterTokens?: number }>(
                'agent:compact',
                activeId
              )
              const saved =
                r.ok && r.beforeTokens != null && r.afterTokens != null
                  ? ` Freed about ${Math.max(0, r.beforeTokens - r.afterTokens).toLocaleString()} tokens.`
                  : ''
              toast(`${r.message}${saved}`, r.ok ? 'success' : 'info')
            }}
            title="Summarise older turns to free context"
            data-testid="compact-session"
          >
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

        <div className="messages" data-testid="agent-messages" ref={scrollRef}>
          <div className="messages-content" ref={contentRef}>
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
              <MessageRow
                role={m.role}
                key={m.id}
                actions={
                  editing === m.id ? undefined : (
                    <MessageActions
                      onCopy={() => void navigator.clipboard.writeText(m.content)}
                      /*
                       * Not while the model is working, and not on a turn that is only a tool
                       * call. Editing rewrites the history the running turn is reading from, and
                       * a turn with no words in it has nothing to rewrite.
                       */
                      onEdit={busy || !(m.content || m.reasoning) ? undefined : () => setEditing(m.id)}
                      /*
                       * Deleting mid-turn would race the running turn's own writes back to
                       * storage once it finishes, so it waits for the same quiet moment editing
                       * does.
                       */
                      onDelete={busy ? undefined : () => void deleteMessage(m.id)}
                    />
                  )
                }
              >
                {editing === m.id ? (
                <MessageEditor
                  content={m.content}
                  /* Present, and separately editable, only where the model actually thought. */
                  reasoning={m.role === 'assistant' ? (m.reasoning ?? '') : undefined}
                  onSave={(content, reasoning) => void saveEdit(m.id, content, m.role, reasoning)}
                  onCancel={() => setEditing(null)}
                  /*
                   * The editor can outlive the moment it was safe to open: a turn can start
                   * elsewhere in this session while it sits open, or the model can be unloaded
                   * from under a rewind. Re-evaluated on every render rather than fixed at open
                   * time, so Save disables itself the instant either happens.
                   */
                  disabled={busy || (m.role === 'user' && !loaded)}
                  disabledReason={
                    busy
                      ? 'The model is busy — wait for it to finish first.'
                      : m.role === 'user' && !loaded
                        ? 'Load a model before rewinding a message.'
                        : undefined
                  }
                />
              ) : m.role === 'assistant' ? (
                <>
                  {m.reasoning && <ThinkingBlock text={m.reasoning} />}
                  {/* A turn can be reasoning only — the model thought, then called a tool
                      without a word of prose. Rendering an empty body leaves a stray gap. */}
                  {m.content && <Markdown source={m.content} />}
                </>
              ) : (
                <>
                  {/* The names are dropped from the text once the files themselves are shown. */}
                  <div className="body">{stripAttachmentLine(m.content, !!m.attachments?.length)}</div>
                  {m.attachments?.length ? <MessageMedia items={m.attachments} /> : null}
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

          {/*
            * Where the answer stopped, and an offer to carry on from it.
            *
            * Under the last message rather than beside the composer, because that is what it
            * acts on — and it goes before the live rows below, which only exist while a turn is
            * running and the offer is therefore hidden.
            */}
          {canContinue && (
            <div className="msg-aside continue-row">
              <button className="continue-button" onClick={() => void continueAnswer()} data-testid="agent-continue">
                <Icon name="resume" size={13} />
                Continue this answer
              </button>
            </div>
          )}

          {unsavedCalls.map((entry) => (
            <ToolCard key={entry.call.id} call={entry.call} result={entry.result} />
          ))}

          {/*
            * Calls still being written, in the position their finished cards will take.
            *
            * After the completed ones, because that is the order they happened in — the model
            * finishes one call before starting the next, and this is the one it is on now.
            */}
          {pendingCalls.map((c) => (
            <PendingToolCall key={c.index} name={c.name} args={c.args} />
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
          {/*
            * Progress wins over the other conditions rather than joining them.
            *
            * An agent turn is many requests, not one: every step after a tool result re-reads a
            * prompt that has grown by whatever that tool returned. By then `partial` and the
            * tool list are both non-empty, so gating on those — as the dots do — would hide the
            * bar for exactly the steps where the wait is longest.
            */}
          {running && (progress || (!partial && !reasoning && !unsavedCalls.length && !pendingCalls.length && !ultra.length)) && (
            <MessageRow role="assistant">
              {/*
                * Preparing an attachment comes before the model is given anything at all, so it
                * outranks the prompt readout: while ffmpeg is sampling a video there is no prompt
                * yet to be making progress through.
                */}
              {media ? (
                <MediaStage {...media} />
              ) : progress ? (
                <PromptProgress {...progress} />
              ) : (
                <div className="thinking">
                  <span className="dot" />
                  <span className="dot" />
                  <span className="dot" />
                </div>
              )}
            </MessageRow>
          )}
          </div>
        </div>

        <div className="composer">
          <JumpToLatest show={detached} onClick={jumpToLatest} />
          {compacting && <CompactingNotice since={compacting.since} automatic={compacting.automatic} />}
          <AttachmentBar items={attachments.items} onRemove={attachments.remove} disabled={busy} />
          <div className="composer-shell">
            <button
              className="attach-button"
              onClick={() => void attachments.pick()}
              disabled={busy || attachments.busy}
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
              disabled={!loaded || busy}
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
              disabled={!loaded || (!running && (!!compacting || (!input.trim() && !attachments.items.length)))}
              title={running ? 'Stop the agent' : 'Send  (Enter)'}
              aria-label={running ? 'Stop' : 'Send'}
              data-testid="agent-send"
            >
              <Icon name={running ? 'stop' : 'send'} size={15} />
            </button>
          </div>
          <div className="composer-meta">
            {ctx && <ContextMeter used={ctx.used} max={ctx.max} />}
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
