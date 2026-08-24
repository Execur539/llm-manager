import { useEffect, useRef, useState } from 'react'
import type { AgentMessage } from '@shared/types'
import { invoke, fmtRelative } from '../lib/api'
import { select, setRunning, takePending, dropPending, useStream, clearFor } from '../lib/store'
import type { LoadedModel } from '../App'
import ConversationList, { type ChatSummary } from '../components/ConversationList'
import Markdown from '../components/Markdown'

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

  // Streaming and selection both live in the store, so leaving this page mid-response no longer
  // discards the text or forgets which conversation was open.
  const stream = useStream()
  const activeId = stream.selection.chat
  const partial = activeId ? (stream.partial[activeId] ?? '') : ''
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
    if (!input.trim() || busy || !loaded) return

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
      await invoke('chat:send', chatId, text, [], collectionId || undefined)
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

      <div className="chat">
        <div className="row head" style={{ flexWrap: 'wrap' }}>
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
            <button onClick={() => void invoke('chat:export', activeId, 'md')} title="Export this conversation">
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
            <div className="empty">
              {collectionId
                ? 'Ask a question about the selected documents.'
                : 'Start a conversation. Attach a document collection to ground answers in your own files.'}
            </div>
          )}

          {messages.map((m) => (
            <div className="msg" key={m.id}>
              <div className="who">{m.role}</div>
              {m.role === 'assistant' ? (
                <Markdown source={m.content} />
              ) : (
                <div className="body">{m.content}</div>
              )}
            </div>
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
          {busy && !partial && (
            <div className="msg">
              <div className="who">assistant</div>
              <div className="body dim">Thinking…</div>
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
            placeholder={loaded ? 'Message…  (Enter to send, Shift+Enter for a newline)' : 'Load a model first'}
            disabled={!loaded || busy}
            data-testid="chat-input"
          />
          <button className="primary" onClick={() => void send()} disabled={!loaded || busy} data-testid="chat-send">
            {busy ? '…' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  )
}
