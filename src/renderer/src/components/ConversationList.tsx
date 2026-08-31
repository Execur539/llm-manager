import { useEffect, useRef, useState } from 'react'
import { closeRail, useStream } from '../lib/store'
import { DRAWER_QUERY, useMediaQuery } from '../lib/media'
import { fmtRelative } from '../lib/api'
import Icon from './Icon'

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

  const { railOpen } = useStream()
  // Below this width the rail is an overlay, so when closed it must be neither visible nor
  // reachable: an off-screen drawer whose buttons still take focus is a keyboard trap.
  const isDrawer = useMediaQuery(DRAWER_QUERY)
  const hidden = isDrawer && !railOpen

  const commitRename = (id: string): void => {
    const title = draft.trim()
    if (title) onRename(id, title)
    setEditingId(null)
  }

  return (
    <>
      {/* Only rendered as a real backdrop on narrow viewports; CSS hides it otherwise. */}
      <div
        className={`rail-backdrop${railOpen ? ' open' : ''}`}
        onClick={closeRail}
        aria-hidden="true"
      />
      <aside
        className={`side-list${railOpen ? ' open' : ''}`}
        data-testid="conversation-list"
        aria-hidden={hidden || undefined}
        {...(hidden ? { inert: '' } : {})}
      >
      <button className="primary new-button" onClick={() => {
          onNew()
          closeRail()
        }}
        data-testid="new-conversation">
        {newLabel}
      </button>

      {items.map((item) => {
        const isEditing = editingId === item.id
        const isConfirming = confirmingId === item.id

        return (
          <div
            key={item.id}
            className={`side-item ${activeId === item.id ? 'active' : ''}`}
            onClick={() => {
              if (isEditing || isConfirming) return
              onOpen(item.id)
              closeRail()
            }}
            data-testid="conversation-item"
            data-id={item.id}
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

            {/*
              * A floating cluster over the row rather than a third line inside it.
              *
              * These used to be text buttons that went from display:none to display:flex on
              * hover, which grew the card by a whole row under the cursor and shoved every
              * conversation below it down the list. Moving the mouse down the rail made the
              * thing you were aiming at walk away from you.
              *
              * Absolutely positioned, the row never changes size, so nothing moves. Icons keep
              * the cluster narrow enough to sit over the title without a scrim, and the label
              * survives for assistive tech as aria-label.
              */}
            {!isEditing && (
              <div className={`side-item-actions${isConfirming ? ' confirming' : ''}`}>
                {isConfirming ? (
                  <>
                    <button
                      className="row-action danger"
                      title="Confirm delete"
                      aria-label={`Confirm deleting ${item.title}`}
                      onClick={(e) => {
                        e.stopPropagation()
                        setConfirmingId(null)
                        onDelete(item.id)
                      }}
                      data-testid="confirm-delete"
                    >
                      <Icon name="check" size={13} />
                    </button>
                    <button
                      className="row-action"
                      title="Cancel"
                      aria-label="Cancel delete"
                      onClick={(e) => {
                        e.stopPropagation()
                        setConfirmingId(null)
                      }}
                    >
                      <Icon name="close" size={13} />
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      className="row-action"
                      title="Rename"
                      aria-label={`Rename ${item.title}`}
                      onClick={(e) => {
                        e.stopPropagation()
                        setDraft(item.title)
                        setEditingId(item.id)
                      }}
                      data-testid="rename-conversation"
                    >
                      <Icon name="pencil" size={13} />
                    </button>
                    <button
                      className="row-action danger"
                      title="Delete"
                      aria-label={`Delete ${item.title}`}
                      onClick={(e) => {
                        e.stopPropagation()
                        setConfirmingId(item.id)
                      }}
                      data-testid="delete-conversation"
                    >
                      <Icon name="trash" size={13} />
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
    </>
  )
}
