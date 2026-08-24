import { useEffect, useRef, useState } from 'react'
import { fmtRelative } from '../lib/api'

export interface ChatSummary {
  id: string
  title: string
  cwd?: string | null
  updatedAt: number
  messageCount: number
}

/**
 * Sidebar list shared by Chat and Agent.
 *
 * Both views previously listed conversations with no way to delete or rename one — the handlers
 * existed in the main process with nothing calling them. Keeping this in one component means the
 * two views cannot drift apart again.
 */
export default function ConversationList({
  items,
  activeId,
  newLabel,
  emptyLabel,
  runningIds,
  onNew,
  onOpen,
  onDelete,
  onRename
}: {
  items: ChatSummary[]
  activeId: string | null
  newLabel: string
  emptyLabel: string
  runningIds: Record<string, boolean>
  onNew: () => void
  onOpen: (id: string) => void
  onDelete: (id: string) => void
  onRename: (id: string, title: string) => void
}): JSX.Element {
  const [confirmingId, setConfirmingId] = useState<string | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (editingId) inputRef.current?.select()
  }, [editingId])

  // A pending delete confirmation should not persist once attention moves elsewhere.
  useEffect(() => {
    if (!confirmingId) return
    const timer = setTimeout(() => setConfirmingId(null), 5000)
    return () => clearTimeout(timer)
  }, [confirmingId])

  const commitRename = (id: string): void => {
    const title = draft.trim()
    if (title) onRename(id, title)
    setEditingId(null)
  }

  return (
    <aside className="side-list" data-testid="conversation-list">
      <button className="primary new-button" onClick={onNew} data-testid="new-conversation">
        {newLabel}
      </button>

      {items.map((item) => {
        const isEditing = editingId === item.id
        const isConfirming = confirmingId === item.id

        return (
          <div
            key={item.id}
            className={`side-item ${activeId === item.id ? 'active' : ''}`}
            onClick={() => !isEditing && !isConfirming && onOpen(item.id)}
            data-testid="conversation-item"
            title={item.title}
          >
            {isEditing ? (
              <input
                ref={inputRef}
                className="rename-input"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onClick={(e) => e.stopPropagation()}
                onBlur={() => commitRename(item.id)}
                onKeyDown={(e) => {
                  e.stopPropagation()
                  if (e.key === 'Enter') commitRename(item.id)
                  if (e.key === 'Escape') setEditingId(null)
                }}
                data-testid="rename-input"
              />
            ) : (
              <>
                <div className="side-item-row">
                  <span className="truncate">{item.title}</span>
                  {runningIds[item.id] && <span className="dot-pulse" title="Response in progress" />}
                </div>
                <div className="faint side-item-meta">
                  {item.messageCount} {item.messageCount === 1 ? 'message' : 'messages'} · {fmtRelative(item.updatedAt)}
                </div>
              </>
            )}

            {!isEditing && (
              <div className="side-item-actions">
                {isConfirming ? (
                  <>
                    <button
                      className="tiny danger"
                      onClick={(e) => {
                        e.stopPropagation()
                        setConfirmingId(null)
                        onDelete(item.id)
                      }}
                      data-testid="confirm-delete"
                    >
                      Delete
                    </button>
                    <button
                      className="tiny"
                      onClick={(e) => {
                        e.stopPropagation()
                        setConfirmingId(null)
                      }}
                    >
                      Cancel
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      className="tiny"
                      title="Rename"
                      onClick={(e) => {
                        e.stopPropagation()
                        setDraft(item.title)
                        setEditingId(item.id)
                      }}
                      data-testid="rename-conversation"
                    >
                      Rename
                    </button>
                    <button
                      className="tiny"
                      title="Delete"
                      onClick={(e) => {
                        e.stopPropagation()
                        setConfirmingId(item.id)
                      }}
                      data-testid="delete-conversation"
                    >
                      Delete
                    </button>
                  </>
                )}
              </div>
            )}
          </div>
        )
      })}

      {!items.length && <div className="empty small">{emptyLabel}</div>}
    </aside>
  )
}
