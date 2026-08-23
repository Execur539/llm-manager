import { useEffect, useRef, useState } from 'react'
import type { AgentMessage } from '@shared/types'
import { invoke, on, fmtRelative } from '../lib/api'
import type { LoadedModel } from '../App'

interface ChatSummary {
  id: string
  title: string
  updatedAt: number
  messageCount: number
}

interface Collection {
  id: string
  name: string
  documents: number
}

export default function ChatView({ loaded }: { loaded: LoadedModel | null }): JSX.Element {
  const [chats, setChats] = useState<ChatSummary[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<AgentMessage[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notes, setNotes] = useState<string[]>([])
  const [collections, setCollections] = useState<Collection[]>([])
  const [collectionId, setCollectionId] = useState<string>('')
  const endRef = useRef<HTMLDivElement>(null)

  const refreshChats = async (): Promise<void> => {
    setChats(await invoke<ChatSummary[]>('chat:list', 'chat'))
  }

  useEffect(() => {
    void refreshChats()
    void invoke<Collection[]>('rag:collections').then(setCollections).catch(() => undefined)

    const offs = [
      on<{ chatId: string; text: string }>('chat:delta', (d) => setStreaming((s) => s + d.text)),
      on<{ chatId: string; message: AgentMessage }>('chat:message', (d) => {
        setStreaming('')
        setMessages((prev) => (prev.some((m) => m.id === d.message.id) ? prev : [...prev, d.message]))
      }),
      on<string[]>('chat:notes', setNotes)
    ]
    return () => offs.forEach((off) => off())
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streaming])

  const openChat = async (id: string): Promise<void> => {
    setActiveId(id)
    setNotes([])
    const session = await invoke<{ messages: AgentMessage[] } | null>('chat:load', id)
    setMessages(session?.messages ?? [])
  }

  const newChat = async (): Promise<void> => {
    const chat = await invoke<ChatSummary>('chat:create', { kind: 'chat', modelId: loaded?.modelId ?? null })
    await refreshChats()
    setActiveId(chat.id)
    setMessages([])
  }

  const send = async (): Promise<void> => {
    if (!input.trim() || busy || !loaded) return
    let chatId = activeId
    if (!chatId) {
      const chat = await invoke<ChatSummary>('chat:create', { kind: 'chat', modelId: loaded.modelId })
      chatId = chat.id
      setActiveId(chatId)
    }

    const text = input
    setInput('')
    setBusy(true)
    setError(null)
    try {
      await invoke('chat:send', chatId, text, [], collectionId || undefined)
      await refreshChats()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="split">
      <aside className="side-list">
        <button className="primary" style={{ width: '100%', marginBottom: 10 }} onClick={() => void newChat()}>
          New chat
        </button>
        {chats.map((c) => (
          <div
            key={c.id}
            className={`side-item ${activeId === c.id ? 'active' : ''}`}
            onClick={() => void openChat(c.id)}
          >
            <div className="truncate">{c.title}</div>
            <div className="faint" style={{ fontSize: 10 }}>
              {c.messageCount} messages · {fmtRelative(c.updatedAt)}
            </div>
          </div>
        ))}
        {!chats.length && <div className="empty" style={{ fontSize: 12 }}>No chats yet.</div>}
      </aside>

      <div className="chat">
        <div className="row" style={{ marginBottom: 10, flexWrap: 'wrap' }}>
          <h1 style={{ marginRight: 'auto' }}>Chat</h1>
          {collections.length > 0 && (
            <select value={collectionId} onChange={(e) => setCollectionId(e.target.value)}>
              <option value="">No documents</option>
              {collections.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} ({c.documents})
                </option>
              ))}
            </select>
          )}
          {activeId && (
            <button onClick={() => void invoke('chat:export', activeId, 'md')}>Export</button>
          )}
        </div>

        {!loaded && (
          <div className="card note">No model is loaded. Load one from <strong>My models</strong> first.</div>
        )}

        {notes.map((n, i) => (
          <div className="card note" key={i}>{n}</div>
        ))}

        {error && (
          <div className="card" style={{ borderColor: '#5c2626' }}>
            <span className="badge bad">error</span> {error}
          </div>
        )}

        <div className="messages">
          {!messages.length && !streaming && (
            <div className="empty">
              {collectionId
                ? 'Ask a question about the selected documents.'
                : 'Start a conversation. Attach a document collection to ground answers in your own files.'}
            </div>
          )}

          {messages.map((m) => (
            <div className="msg" key={m.id}>
              <div className="who">{m.role}</div>
              <div className="body">{m.content}</div>
            </div>
          ))}

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
            placeholder={loaded ? 'Message…  (Enter to send, Shift+Enter for a newline)' : 'Load a model first'}
            disabled={!loaded || busy}
          />
          <button className="primary" onClick={() => void send()} disabled={!loaded || busy}>
            {busy ? '…' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  )
}
