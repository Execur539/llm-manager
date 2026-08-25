import { useEffect, useRef, useState } from 'react'
import type { AgentMessage } from '@shared/types'
import { invoke, fmtRelative } from '../lib/api'
import { select, setRunning, takePending, dropPending, useStream, clearFor, toast, setReasoning } from '../lib/store'
import type { LoadedModel } from '../App'
import ConversationList, { type ChatSummary } from '../components/ConversationList'
import Icon from '../components/Icon'
import Markdown from '../components/Markdown'
import MessageRow from '../components/MessageRow'
import ThinkingBlock from '../components/ThinkingBlock'
import RailToggle from '../components/RailToggle'
import ReasoningControl from '../components/ReasoningControl'
import { AttachmentBar, DropZone, useAttachments } from '../components/Attachments'
import { Spinner } from '../components/Spinner'
import EmptyState from '../components/EmptyState'

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
  const endRef = useRef<HTMLDivElement>(null)
  const attachments = useAttachments()

  // Streaming and selection both live in the store, so leaving this page mid-response no longer
  // discards the text or forgets which conversation was open.
  const stream = useStream()
  const activeId = stream.selection.chat
  const partial = activeId ? (stream.partial[activeId] ?? '') : ''
  const reasoning = activeId ? (stream.reasoningPartial[activeId] ?? '') : ''
  const busy = activeId ? !!stream.running[activeId] : false

  const refreshChats = async (): Promise<void> => {
    setChats(await invoke<ChatSummary[]>('chat:list', 'chat'))
  }

  useEffect(() => {
    void refreshChats()
    void invoke<Collection[]>('rag:collections').then(setCollections).catch(() => undefined)
  }, [])

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
      const session = await invoke<{ messages: AgentMessage[] } | null>('chat:load', activeId)
      if (cancelled) return
      setMessages(session?.messages ?? [])
      dropPending(activeId)
    })()
    return () => {
      cancelled = true
    }
  }, [activeId])

  // Absorb anything that arrived while this view was unmounted.
  useEffect(() => {
    if (!activeId) return
    const pending = takePending(activeId)
    if (pending.length) {
      setMessages((prev) => {
        const seen = new Set(prev.map((m) => m.id))
        return [...prev, ...pending.filter((m) => !seen.has(m.id))]
      })
    }
  }, [activeId, stream.pending])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, partial])

  const openChat = (id: string): void => {
    setError(null)
    select('chat', id)
  }

  const newChat = async (): Promise<void> => {
    const chat = await invoke<ChatSummary>('chat:create', { kind: 'chat', modelId: loaded?.modelId ?? null })
    await refreshChats()
    setMessages([])
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

  const send = async (): Promise<void> => {
    if ((!input.trim() && !attachments.items.length) || busy || !loaded) return

    let chatId = activeId
    if (!chatId) {
      const chat = await invoke<ChatSummary>('chat:create', { kind: 'chat', modelId: loaded.modelId })
      chatId = chat.id
      setMessages([])
      select('chat', chatId)
    }

    const text = input
    setInput('')
    setError(null)
    setRunning(chatId, true)

    try {
      const files = attachments.items.map((a) => a.path)
      attachments.clear()
      await invoke('chat:send', chatId, text, files, collectionId || undefined, stream.reasoning[chatId] ?? null)
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

      <DropZone onFiles={(f) => void attachments.addFiles(f)} disabled={busy}>
      <div className="chat">
        <div className="row head chat-head">
          <RailToggle />
          <h1 style={{ marginRight: 'auto' }}>Chat</h1>
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
              onClick={() => {
                // Fire-and-forget used to mean the button looked broken: a dialog appeared, then
                // nothing. Report the written path, or the failure, either way.
                void invoke<string | null>('chat:export', activeId, 'md')
                  .then((file) => {
                    if (file) toast(`Exported to ${file}`, 'success', file)
                  })
                  .catch((err: unknown) => toast(`Export failed: ${String(err)}`, 'error'))
              }}
              title="Export this conversation"
              data-testid="export-chat"
            >
              Export
            </button>
          )}
        </div>

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

        <div className="messages" data-testid="chat-messages">
          {!messages.length && !partial && (
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

          {messages.map((m) => (
            <MessageRow role={m.role} key={m.id}>
              {m.role === 'assistant' ? (
                <>
                  {m.reasoning && <ThinkingBlock text={m.reasoning} />}
                  {/* Kept in step with Agent: a reasoning-only turn renders no empty body. */}
                  {m.content && <Markdown source={m.content} />}
                </>
              ) : (
                <div className="body">{m.content}</div>
              )}
            </MessageRow>
          ))}

          {(partial || reasoning) && (
            <MessageRow role="assistant" testId="streaming-message">
              {reasoning && <ThinkingBlock text={reasoning} streaming answerStarted={!!partial} />}
              <div className="body streaming">
                <Markdown source={partial} caret />
              </div>
            </MessageRow>
          )}
          {busy && !partial && !reasoning && (
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
              placeholder={loaded ? 'Message…' : 'Load a model first'}
              disabled={!loaded || busy}
              rows={1}
              data-testid="chat-input"
            />
            <button
              className="send-button"
              onClick={() => void send()}
              disabled={!loaded || busy || (!input.trim() && !attachments.items.length)}
              title={busy ? 'Waiting for the model' : 'Send  (Enter)'}
              aria-label="Send"
              data-testid="chat-send"
            >
              <Icon name="send" size={15} />
            </button>
          </div>
          <div className="composer-meta">
            <ReasoningControl
              support={loaded?.caps?.reasoning}
              value={activeId ? (stream.reasoning[activeId] ?? null) : null}
              onChange={(next) => activeId && setReasoning(activeId, next)}
              disabled={busy}
            />
            <div className="composer-hint">Enter to send · Shift+Enter for a newline</div>
          </div>
        </div>
      </div>
      </DropZone>
    </div>
  )
}
