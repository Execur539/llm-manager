import { useEffect, useState } from 'react'
import { fmtBytes, invoke, on } from '../lib/api'

interface Collection {
  id: string
  name: string
  documents: number
}

interface DocumentRecord {
  id: string
  filename: string
  bytes: number
  chunks: number
}

export default function Documents(): JSX.Element {
  const [collections, setCollections] = useState<Collection[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [documents, setDocuments] = useState<DocumentRecord[]>([])
  const [newName, setNewName] = useState('')
  const [progress, setProgress] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [hits, setHits] = useState<{ filename: string; text: string; score: number }[]>([])

  const refresh = async (): Promise<void> => {
    setCollections(await invoke<Collection[]>('rag:collections'))
  }

  useEffect(() => {
    void refresh()
    const off = on<{ file?: string; phase: string; error?: string }>('rag:progress', (p) => {
      if (p.phase === 'done') {
        setProgress(null)
        void refresh()
        if (activeId) void loadDocs(activeId)
      } else if (p.phase === 'failed') {
        setError(`${p.file}: ${p.error}`)
      } else {
        setProgress(`Embedding ${p.file}…`)
      }
    })
    return off
  }, [activeId])

  const loadDocs = async (collectionId: string): Promise<void> => {
    setActiveId(collectionId)
    setDocuments(await invoke<DocumentRecord[]>('rag:documents', { collectionId }))
  }

  const create = async (): Promise<void> => {
    if (!newName.trim()) return
    await invoke('rag:create-collection', newName.trim())
    setNewName('')
    await refresh()
  }

  const addFiles = async (): Promise<void> => {
    if (!activeId) return
    setError(null)
    setProgress('Choosing files…')
    try {
      await invoke('rag:ingest', { collectionId: activeId })
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setProgress(null)
      await loadDocs(activeId)
      await refresh()
    }
  }

  const search = async (): Promise<void> => {
    if (!query.trim() || !activeId) return
    setError(null)
    try {
      setHits(await invoke('rag:retrieve', query, { collectionId: activeId }))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <>
      <h1>Documents</h1>
      <p className="subtitle">
        Build reusable collections and attach them to any chat, or drop files straight into a conversation for
        one-offs. Everything is embedded locally with the bundled model.
      </p>

      {error && (
        <div className="card" style={{ borderColor: '#5c2626' }}>
          <span className="badge bad">error</span> {error}
        </div>
      )}
      {progress && <div className="card note">{progress}</div>}

      <div className="split">
        <aside className="side-list">
          <div className="row" style={{ marginBottom: 10 }}>
            <input
              type="text"
              placeholder="New collection…"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && void create()}
              style={{ flex: 1 }}
            />
            <button onClick={() => void create()}>+</button>
          </div>
          {collections.map((c) => (
            <div
              key={c.id}
              className={`side-item ${activeId === c.id ? 'active' : ''}`}
              onClick={() => void loadDocs(c.id)}
            >
              <div className="truncate">{c.name}</div>
              <div className="faint" style={{ fontSize: 10 }}>{c.documents} documents</div>
            </div>
          ))}
          {!collections.length && <div className="empty" style={{ fontSize: 12 }}>No collections yet.</div>}
        </aside>

        <div style={{ flex: 1, minWidth: 0 }}>
          {!activeId && <div className="empty">Select or create a collection.</div>}

          {activeId && (
            <>
              <div className="row" style={{ marginBottom: 12 }}>
                <button className="primary" onClick={() => void addFiles()}>Add documents…</button>
                <button
                  className="danger"
                  onClick={async () => {
                    if (!confirm('Delete this collection and its embeddings?')) return
                    await invoke('rag:delete-collection', activeId)
                    setActiveId(null)
                    setDocuments([])
                    await refresh()
                  }}
                >
                  Delete collection
                </button>
              </div>

              <div className="list">
                {documents.map((d) => (
                  <div className="card row-card" key={d.id}>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div className="truncate">{d.filename}</div>
                      <div className="faint" style={{ fontSize: 11 }}>
                        {fmtBytes(d.bytes)} · {d.chunks} chunks
                      </div>
                    </div>
                    <button
                      className="danger"
                      onClick={async () => {
                        await invoke('rag:delete-document', d.id)
                        await loadDocs(activeId)
                      }}
                    >
                      Remove
                    </button>
                  </div>
                ))}
                {!documents.length && <div className="empty">No documents in this collection yet.</div>}
              </div>

              {documents.length > 0 && (
                <>
                  <h3 style={{ fontSize: 13, marginTop: 22 }}>Test retrieval</h3>
                  <div className="row" style={{ marginBottom: 10 }}>
                    <input
                      type="text"
                      placeholder="Ask something to see which chunks come back…"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && void search()}
                      style={{ flex: 1 }}
                    />
                    <button onClick={() => void search()}>Search</button>
                  </div>
                  {hits.map((h, i) => (
                    <div className="card" key={i}>
                      <div className="row" style={{ justifyContent: 'space-between' }}>
                        <span className="mono" style={{ fontSize: 11 }}>{h.filename}</span>
                        <span className="badge">{h.score.toFixed(3)}</span>
                      </div>
                      <div className="dim" style={{ fontSize: 12, marginTop: 6 }}>{h.text.slice(0, 400)}…</div>
                    </div>
                  ))}
                </>
              )}
            </>
          )}
        </div>
      </div>
    </>
  )
}
