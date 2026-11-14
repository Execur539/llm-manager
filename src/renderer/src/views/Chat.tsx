import { Fragment, useEffect, useRef, useState } from 'react'
import type { AgentMessage } from '@shared/types'
import { invoke, on, fmtRelative } from '../lib/api'
import {
  select,
  setRunning,
  takePending,
  useStream,
  clearFor,
  toast,
  setReasoning,
  adoptReasoning,
  seedContext,
  seedSummary,
  DRAFT_CHAT
} from '../lib/store'
import type { LoadedModel } from '../App'
import ConversationList, { type ChatSummary } from '../components/ConversationList'
import Icon from '../components/Icon'
import Markdown from '../components/Markdown'
import MessageRow from '../components/MessageRow'
import MessageMedia, { stripAttachmentLine } from '../components/MessageMedia'
import ThinkingBlock from '../components/ThinkingBlock'
import UltraSamples from '../components/UltraSamples'
import RailToggle from '../components/RailToggle'
import { Toolbar } from '../lib/shell'
import ReasoningControl, { sendableChoice } from '../components/ReasoningControl'
import { AttachmentBar, DropZone, useAttachments } from '../components/Attachments'
import { Spinner } from '../components/Spinner'
import PromptProgress from '../components/PromptProgress'
import MediaStage from '../components/MediaStage'
import ContextMeter from '../components/ContextMeter'
import CompactingNotice from '../components/CompactingNotice'
import { SaveStateButton, saveMarks, useSaveCoverage, withSavePoint } from '../components/SaveState'
import { LockedPanel, ProtectButton } from '../components/Protect'
import ConfirmDialog from '../components/ConfirmDialog'
import JumpToLatest from '../components/JumpToLatest'
import { useStickToBottom } from '../lib/useStickToBottom'
import EmptyState from '../components/EmptyState'
import ToolCard from '../components/ToolCard'
import PendingToolCall from '../components/PendingToolCall'
import ParametersPanel, { ParamsToggle, useParamsPanel } from '../components/ParametersPanel'

interface Collection {
  id: string
  name: string
  documents: number
}

export default function ChatView({ loaded }: { loaded: LoadedModel | null }): JSX.Element {
  const [chats, setChats] = useState<ChatSummary[]>([])
  const [messages, setMessages] = useState<AgentMessage[]>([])
  const [input, setInput] = useState('')
  const [collections, setCollections] = useState<Collection[]>([])
  const [collectionId, setCollectionId] = useState('')
  const [error, setError] = useState<string | null>(null)
  const attachments = useAttachments()
  const [paramsOpen, setParamsOpen] = useParamsPanel()

  // Streaming and selection both live in the store, so leaving this page mid-response no longer
  // discards the text or forgets which conversation was open.
  const stream = useStream()
  const activeId = stream.selection.chat
  const partial = activeId ? (stream.partial[activeId] ?? '') : ''
  const reasoning = activeId ? (stream.reasoningPartial[activeId] ?? '') : ''
  const busy = activeId ? !!stream.running[activeId] : false
  const progress = activeId ? (stream.promptProgress[activeId] ?? null) : null
  const media = activeId ? (stream.mediaStage[activeId] ?? null) : null
  // The scratchpad tool the model is running right now, if any.
  const toolRunning = activeId ? (stream.chatToolRunning[activeId] ?? null) : null
  // Scratchpad calls the model is still writing.
  const writing = activeId ? (stream.pendingCalls[activeId] ?? []) : []
  const ctx = activeId ? (stream.context[activeId] ?? null) : null
  const compacting = activeId ? (stream.compacting[activeId] ?? null) : null
  const summaryUpto = activeId ? (stream.summaryUpto[activeId] ?? null) : null
  const activeChat = chats.find((c) => c.id === activeId)
  // Protected with a password and not unlocked: nothing of it can be shown or sent to.
  const sealed = !!activeChat?.locked
  // Mid-compaction the conversation takes no input, exactly as while a reply is being written.
  const locked = busy || !!compacting || sealed
  const [reloadTick, setReloadTick] = useState(0)
  const [confirmExport, setConfirmExport] = useState(false)
  // Which of the conversation the saved model state holds, drawn as an edge on each message.
  const saveKey = activeId ? `chat:${activeId}` : null
  const coverage = useSaveCoverage(saveKey, messages)
  const savingNow = saveKey ? stream.savingStates[saveKey] : undefined
  const marks = saveMarks(messages, coverage, savingNow, partial || reasoning ? { content: partial, reasoning } : null)
  const allSaved =
    !!coverage?.through && !coverage.otherModel && coverage.through === messages.at(-1)?.id && !partial && !reasoning && !busy
  const live = withSavePoint(marks.streaming, partial, reasoning)
  // Usable before the first message: the choice is held against a draft key and adopted by
  // whatever conversation the message goes on to create.
  const effortId = activeId ?? DRAFT_CHAT
  const ultra = activeId ? (stream.ultra[activeId] ?? []) : []
  const synthesising = activeId ? !!stream.ultraSynthesising[activeId] : false

  // Keyed on the conversation, so switching to another one opens at its latest message.
  const { scrollRef, contentRef, detached, jumpToLatest } = useStickToBottom(activeId)

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

  const refreshChats = async (): Promise<void> => {
    setChats(await invoke<ChatSummary[]>('chat:list', 'chat'))
  }

  useEffect(() => {
    void refreshChats()
    void invoke<Collection[]>('rag:collections').then(setCollections).catch(() => undefined)
  }, [])

  // The model names chats in the background once a reply is in, and renames them later.
  useEffect(() => on('chat:titled', () => void refreshChats()), [])

  // Protected, unlocked or locked: the list says so, and the open conversation is read again.
  useEffect(
    () =>
      on<{ kind: string; id: string }>('protect:changed', (d) => {
        void refreshChats()
        if (d.kind === 'chat') setReloadTick((n) => n + 1)
        if (d.kind === 'collection') void invoke<Collection[]>('rag:collections').then(setCollections).catch(() => undefined)
      }),
    []
  )

  // Reload history whenever the selected conversation changes — including on remount, which is
  // what restores the view after navigating away mid-response. The database is authoritative
  // because the main process persists each message before emitting it, so anything buffered
  // while unmounted is already included and can be dropped.
  useEffect(() => {
    if (!activeId) {
      setMessages([])
      return
    }
    let cancelled = false
    void (async () => {
      const session = await invoke<{ messages: AgentMessage[]; contextUsed?: number; summaryUpto?: string } | null>(
        'chat:load',
        activeId
      )
      if (cancelled) return
      setMessages(session?.messages ?? [])
      // So the meter reads correctly on reopening rather than staying blank until the next turn.
      seedContext(activeId, session?.contextUsed, loaded?.plan.contextLength ?? 0)
      seedSummary(activeId, session?.summaryUpto ?? null)
      // Anything that arrived while this was loading joins it, matched by id, rather than being dropped.
      const queued = takePending(activeId)
      if (queued.length) {
        setMessages((prev) => {
          const byId = new Map(prev.map((m) => [m.id, m]))
          for (const m of queued) byId.set(m.id, m)
          return [...byId.values()]
        })
      }
    })()
    return () => {
      cancelled = true
    }
  }, [activeId, loaded?.plan.contextLength, reloadTick])

  // Absorb anything that arrived while this view was unmounted. A repeated id replaces the copy
  // already on screen rather than being dropped, so a message that is revised after it was first
  // shown keeps its place in the transcript and gains whatever the later copy added.
  useEffect(() => {
    if (!activeId) return
    const pending = takePending(activeId)
    if (pending.length) {
      setMessages((prev) => {
        const byId = new Map(prev.map((m) => [m.id, m]))
        for (const m of pending) byId.set(m.id, m)
        return [...byId.values()]
      })
    }
  }, [activeId, stream.pending])

  const openChat = (id: string): void => {
    setError(null)
    select('chat', id)
  }

  const newChat = async (): Promise<void> => {
    const chat = await invoke<ChatSummary>('chat:create', { kind: 'chat', modelId: loaded?.modelId ?? null })
    await refreshChats()
    setMessages([])
    adoptReasoning(DRAFT_CHAT, chat.id)
    select('chat', chat.id)
  }

  const deleteChat = async (id: string): Promise<void> => {
    await invoke('chat:delete', id)
    clearFor(id)
    await refreshChats()
  }

  const renameChat = async (id: string, title: string): Promise<void> => {
    await invoke('chat:rename', id, title)
    await refreshChats()
  }

  const exportActive = (): void => {
    if (!activeId) return
    // Fire-and-forget used to mean the button looked broken: a dialog appeared, then nothing.
    // Report the written path, or the failure, either way.
    void invoke<string | null>('chat:export', activeId, 'md')
      .then((file) => {
        if (file) toast(`Exported to ${file}`, 'success', file)
      })
      .catch((err: unknown) => toast(`Export failed: ${String(err)}`, 'error'))
  }

  const send = async (): Promise<void> => {
    // Not mid-compaction either: a message sent then lands in a history about to be rewritten.
    if ((!input.trim() && !attachments.items.length) || busy || compacting || sealed || !loaded) return

    let chatId = activeId
    if (!chatId) {
      const chat = await invoke<ChatSummary>('chat:create', { kind: 'chat', modelId: loaded.modelId })
      chatId = chat.id
      setMessages([])
      adoptReasoning(DRAFT_CHAT, chatId)
      select('chat', chatId)
    }

    // Read from the render-time snapshot, which is the only place the draft choice exists —
    // adoptReasoning has mutated the store, but `stream` here was captured before it did.
    // Narrowed to what this model can express — see sendableChoice.
    const effort = sendableChoice(loaded?.caps?.reasoning, stream.reasoning[effortId] ?? null)

    const text = input
    setInput('')
    setError(null)
    setRunning(chatId, true)

    try {
      const files = attachments.items.map((a) => a.path)
      attachments.clear()
      await invoke('chat:send', chatId, text, files, collectionId || undefined, effort)
      await refreshChats()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setRunning(chatId, false)
    }
  }

  return (
    <div className="split">
      <ConversationList
        items={chats}
        activeId={activeId}
        newLabel="New chat"
        emptyLabel="No chats yet."
        runningIds={stream.running}
        onNew={() => void newChat()}
        onOpen={(id) => openChat(id)}
        onDelete={(id) => void deleteChat(id)}
        onRename={(id, title) => void renameChat(id, title)}
      />

      <DropZone onFiles={(f) => void attachments.addFiles(f)} disabled={locked}>
      <div className="chat">
        <Toolbar title={chats.find((c) => c.id === activeId)?.title ?? 'Chat'} leading={<RailToggle />}>
          {collections.length > 0 && (
            <select
              value={collectionId}
              onChange={(e) => setCollectionId(e.target.value)}
              title="Ground answers in a document collection"
            >
              <option value="">No documents</option>
              {collections.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} ({c.documents})
                </option>
              ))}
            </select>
          )}
          {activeId && (
            <button
              disabled={!loaded || locked}
              onClick={() => {
                void invoke<{ ok: boolean; message: string; beforeTokens?: number; afterTokens?: number }>(
                  'chat:compact',
                  activeId
                )
                  .then((r) => {
                    const saved =
                      r.ok && r.beforeTokens != null && r.afterTokens != null
                        ? ` Freed about ${Math.max(0, r.beforeTokens - r.afterTokens).toLocaleString()} tokens.`
                        : ''
                    toast(`${r.message}${saved}`, r.ok ? 'success' : 'info')
                  })
                  .catch((err: unknown) =>
                    toast(`Compaction failed: ${err instanceof Error ? err.message : String(err)}`, 'error')
                  )
              }}
              title="Summarise older messages to free context"
              data-testid="compact-chat"
            >
              Compact
            </button>
          )}
          <SaveStateButton stateKey={saveKey} canSave={!!loaded && !sealed && !compacting} allSaved={allSaved} hasMessages={messages.length > 0} />
          <ProtectButton kind="chat" id={activeId} state={activeChat} disabled={busy || !!compacting} />
          {activeId && (
            <button
              disabled={sealed}
              onClick={() => {
                // A protected conversation is exported as plain text, so that is said first.
                if (activeChat?.protected) setConfirmExport(true)
                else exportActive()
              }}
              title="Export this conversation"
              data-testid="export-chat"
            >
              Export
            </button>
          )}
          <ParamsToggle open={paramsOpen} onToggle={() => setParamsOpen(!paramsOpen)} />
        </Toolbar>

        {!loaded && (
          <div className="card note">
            No model is loaded. Load one from <strong>My models</strong> first.
          </div>
        )}
        {error && (
          <div className="card error-card">
            <span className="badge bad">error</span> {error}
          </div>
        )}

        <div className="messages" data-testid="chat-messages" ref={scrollRef}>
          <div className="messages-content" ref={contentRef}>
          {sealed && activeId && <LockedPanel kind="chat" id={activeId} password={activeChat?.password !== false} />}
          {!messages.length && !partial && !sealed && (
            <EmptyState
              icon="chat"
              title={collectionId ? 'Ask about your documents' : 'Start a conversation'}
              body={
                collectionId
                  ? 'Answers will be grounded in the collection selected above.'
                  : 'Attach a document collection to ground answers in your own files.'
              }
            />
          )}

          {messages.map((m) => {
            const mark = marks.byId.get(m.id)
            const shown = withSavePoint(mark, m.content, m.reasoning)
            return (
            <Fragment key={m.id}>
              {/* The model using its scratchpad: a line per call, with the frames it looked at. */}
              {m.role === 'tool' && m.toolCalls?.[0] ? (
                <ToolCard call={m.toolCalls[0]} result={m.toolResult} frames={stream.toolFrames[m.id]} save={mark} />
              ) : (
              <MessageRow role={m.role} save={mark}>
                {m.role === 'assistant' ? (
                  <>
                    {shown.reasoning && <ThinkingBlock text={shown.reasoning} />}
                    {/* Kept in step with Agent: a reasoning-only turn renders no empty body. */}
                    {shown.content && <Markdown source={shown.content} />}
                  </>
                ) : (
                  <>
                    {/* The names are dropped from the text once the files themselves are shown. */}
                    <div className="body">{stripAttachmentLine(m.content, !!m.attachments?.length)}</div>
                    {m.attachments?.length ? <MessageMedia items={m.attachments} /> : null}
                  </>
                )}
              </MessageRow>
              )}
              {/*
                * Where the model's word-for-word view begins.
                *
                * The transcript keeps every message, so without this nothing on screen says the model
                * is working from a summary of everything above — which is exactly when a detail from
                * early on seems to have been forgotten.
                */}
              {m.id === summaryUpto && (
                <div className="summary-divider" role="note" data-testid="summary-divider">
                  The model reads a summary of the messages above
                </div>
              )}
            </Fragment>
            )
          })}

          {/*
            * In the order it happened: the thinking and words of this round, then the calls the
            * model is writing, then the call that is running.
            */}
          {(partial || reasoning || ultra.length > 0) && (
            <MessageRow role="assistant" testId="streaming-message" save={marks.streaming}>
              <UltraSamples samples={ultra} synthesising={synthesising} />
              {reasoning && <ThinkingBlock text={live.reasoning ?? reasoning} streaming answerStarted={!!partial || writing.length > 0} />}
              {partial && (
                <div className="body streaming">
                  <Markdown source={live.content} caret={!writing.length} />
                </div>
              )}
            </MessageRow>
          )}
          {writing.map((c) => (
            <PendingToolCall key={c.index} name={c.name} args={c.args} />
          ))}
          {toolRunning && <ToolCard call={toolRunning} />}
          {busy && !compacting && (progress || (!partial && !reasoning && !ultra.length)) && (
            <MessageRow role="assistant">
              {/*
                * Before the first token there is a real step to report, and until now it looked
                * identical to being stuck. The dots stay as the fallback: a short prompt can be
                * read faster than the server bothers to report, and a server too old to know
                * `return_progress` never reports at all.
                */}
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

        {/*
          * Inside the composer, floated above it.
          *
          * The composer changes height as the textarea grows and as attachments appear, so
          * anchoring the button to the top of that element keeps it clear of the writing area
          * without anyone having to know how tall it currently is.
          */}
        <div className="composer">
          <JumpToLatest show={detached} onClick={jumpToLatest} />
          {compacting && (
            <CompactingNotice
              since={compacting.since}
              automatic={compacting.automatic}
              percent={compacting.percent}
              chunk={compacting.chunk}
              chunks={compacting.chunks}
              stage={compacting.stage}
              helper={compacting.helper}
            />
          )}
          <AttachmentBar items={attachments.items} onRemove={attachments.remove} disabled={locked} />
          <div className="composer-shell">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onPaste={attachments.paste}
              // Clicking in to type is the earliest sign a message is coming: the model starts
              // reading the conversation now, if it is not already holding it, rather than at Send.
              onFocus={() => {
                if (!loaded || busy || !activeId) return
                const effort = sendableChoice(loaded?.caps?.reasoning, stream.reasoning[effortId] ?? null)
                void invoke('chat:warm', activeId, effort).catch(() => undefined)
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  void send()
                }
              }}
              placeholder={loaded ? 'Message…' : 'Load a model first'}
              disabled={!loaded || locked}
              rows={1}
              data-testid="chat-input"
            />
            {/*
              * One button, two jobs. While a turn is in flight the only thing worth doing here
              * is stopping it — a disabled send button says "wait", which is exactly the thing
              * the user is trying not to do.
              */}
            {/* Everything about how the message goes, on one line under what it says. */}
            <div className="composer-bar">
              <button
                className="attach-button"
                onClick={() => void attachments.pick()}
                disabled={locked || attachments.busy}
                title="Attach images, video, audio, or text files — or paste them into the message box"
                aria-label="Attach files"
                data-testid="attach-button"
              >
                {attachments.busy ? <Spinner size={14} /> : <Icon name="plus" size={15} />}
              </button>
              <ReasoningControl
                support={loaded?.caps?.reasoning}
                value={stream.reasoning[effortId] ?? null}
                onChange={(next) => setReasoning(effortId, next)}
                disabled={locked}
                samples={ultraSamples}
                onSamplesChange={setUltraSamples}
              />
              <span className="composer-spacer" />
              {/* Live while the model writes: the server's own measure of how fast. */}
              {busy && activeId && stream.speed[activeId] ? (
                <span className="live-speed" data-testid="live-speed" title="How fast the reply is being written, as the server measures it">
                  {stream.speed[activeId].toFixed(stream.speed[activeId] < 10 ? 1 : 0)} tok/s
                </span>
              ) : null}
              {ctx && <ContextMeter used={ctx.used} max={ctx.max} />}
              <button
                className={`send-button${busy ? ' stopping' : ''}`}
                onClick={() => (busy ? void invoke('chat:stop', activeId) : void send())}
                disabled={!loaded || (!busy && (!!compacting || (!input.trim() && !attachments.items.length)))}
                title={busy ? 'Stop generating' : 'Send  (Enter)'}
                aria-label={busy ? 'Stop' : 'Send'}
                data-testid="chat-send"
              >
                <Icon name={busy ? 'stop' : 'send'} size={15} />
              </button>
            </div>
          </div>
          <div className="composer-hint">Enter to send · Shift+Enter for a newline</div>
        </div>
      </div>
      </DropZone>
      <ConfirmDialog
        open={confirmExport}
        title="Export as plain text?"
        body={
          <p>
            This conversation is protected, but the exported file is not encrypted: any program that can read the folder you
            save it to can read it.
          </p>
        }
        confirmLabel="Export"
        onConfirm={() => {
          setConfirmExport(false)
          exportActive()
        }}
        onCancel={() => setConfirmExport(false)}
      />
      {paramsOpen && (
        <ParametersPanel kind="chat" modelId={loaded?.modelId ?? null} onClose={() => setParamsOpen(false)} />
      )}
    </div>
  )
}
