/**
 * Streaming state that outlives the view.
 *
 * Views are conditionally rendered, so navigating away unmounts them. When streaming state and
 * event subscriptions lived inside a view, leaving the page during a response tore down the
 * listeners and discarded the partial text — the response appeared to vanish, and reappeared
 * only if it happened to finish and persist before you returned.
 *
 * This module subscribes once, at import time, and keeps accumulating regardless of what is
 * mounted. Views read from it and re-render; they no longer own the data.
 */

import { useSyncExternalStore } from 'react'
import type { AgentMessage, PermissionRequest, ToolCall, ToolResult } from '@shared/types'
import { on } from './api'

export interface StreamState {
  /** partial assistant text, keyed by chat/session id */
  partial: Record<string, string>
  /** ids with a turn currently in flight */
  running: Record<string, boolean>
  /** messages that arrived while the view was unmounted, keyed by id */
  pending: Record<string, AgentMessage[]>
  /** last error per id */
  errors: Record<string, string | null>
  /** compaction notices per id */
  notices: Record<string, string | null>
  /** permission prompts are global — only one agent runs at a time */
  permissionQueue: PermissionRequest[]
  /** tool calls seen this turn, so a remount can still render them in order */
  toolCalls: Record<string, { call: ToolCall; result?: ToolResult }[]>
  /**
   * Which conversation each view has open.
   *
   * This has to outlive the view for the same reason the streamed text does: navigating away
   * unmounts the view, and if the selection lived in component state the user would come back
   * to nothing selected — losing sight of a response that is still arriving.
   */
  selection: { chat: string | null; agent: string | null }
}

const state: StreamState = {
  partial: {},
  running: {},
  pending: {},
  errors: {},
  notices: {},
  permissionQueue: [],
  toolCalls: {},
  selection: { chat: null, agent: null }
}

const listeners = new Set<() => void>()

/**
 * A new object identity per change, so useSyncExternalStore sees a difference.
 * Cheap: these maps hold a handful of keys, not message history.
 */
let snapshot: StreamState = { ...state }

function emitChange(): void {
  snapshot = {
    partial: { ...state.partial },
    running: { ...state.running },
    pending: { ...state.pending },
    errors: { ...state.errors },
    notices: { ...state.notices },
    permissionQueue: [...state.permissionQueue],
    toolCalls: { ...state.toolCalls },
    selection: { ...state.selection }
  }
  listeners.forEach((l) => l())
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export function useStream(): StreamState {
  return useSyncExternalStore(subscribe, () => snapshot)
}

// ---------------------------------------------------------------- mutations

export function setRunning(id: string, running: boolean): void {
  state.running[id] = running
  if (running) {
    state.errors[id] = null
    state.partial[id] = ''
    state.toolCalls[id] = []
  }
  emitChange()
}

/** Drain messages that arrived while a view was unmounted. */
export function takePending(id: string): AgentMessage[] {
  const messages = state.pending[id] ?? []
  if (messages.length) {
    state.pending[id] = []
    emitChange()
  }
  return messages
}

export function clearFor(id: string): void {
  if (state.selection.chat === id) state.selection.chat = null
  if (state.selection.agent === id) state.selection.agent = null
  delete state.partial[id]
  delete state.running[id]
  delete state.pending[id]
  delete state.errors[id]
  delete state.notices[id]
  delete state.toolCalls[id]
  emitChange()
}

export function dismissPermission(requestId: string): void {
  state.permissionQueue = state.permissionQueue.filter((r) => r.id !== requestId)
  emitChange()
}

export function clearNotice(id: string): void {
  state.notices[id] = null
  emitChange()
}

/**
 * The id a turn belongs to.
 *
 * The main process reports which chat or session each event concerns. Older events carried no
 * id at all, which is part of why state could not survive a remount; `activeId` is the fallback
 * so a stray event is still attributed somewhere rather than dropped.
 */
let activeId = ''

export function setActiveId(id: string): void {
  activeId = id
}

/** Remember which conversation a view has open, so a remount restores it. */
export function select(kind: 'chat' | 'agent', id: string | null): void {
  state.selection[kind] = id
  if (id) activeId = id
  emitChange()
}

/**
 * Discard buffered messages for an id without applying them.
 *
 * Used after loading history from the database, which is authoritative — the main process
 * persists a message before it emits it, so anything buffered is already in what we just read.
 */
export function dropPending(id: string): void {
  if (state.pending[id]?.length) {
    state.pending[id] = []
    emitChange()
  }
}

// ---------------------------------------------------------------- subscriptions

/** Wired once at import. Nothing here is torn down, because nothing here is owned by a view. */
function wire(): void {
  on<{ chatId: string; text: string }>('chat:delta', (d) => {
    const id = d.chatId || activeId
    state.partial[id] = (state.partial[id] ?? '') + d.text
    emitChange()
  })

  on<{ chatId: string; message: AgentMessage }>('chat:message', (d) => {
    const id = d.chatId || activeId
    state.partial[id] = ''
    state.pending[id] = [...(state.pending[id] ?? []), d.message]
    emitChange()
  })

  on<{ sessionId?: string; text: string } | string>('agent:delta', (payload) => {
    const id = typeof payload === 'string' ? activeId : payload.sessionId || activeId
    const text = typeof payload === 'string' ? payload : payload.text
    state.partial[id] = (state.partial[id] ?? '') + text
    emitChange()
  })

  on<{ sessionId?: string; message?: AgentMessage } | AgentMessage>('agent:message', (payload) => {
    const wrapped = payload as { sessionId?: string; message?: AgentMessage }
    const message = (wrapped.message ?? payload) as AgentMessage
    const id = wrapped.sessionId || activeId
    state.partial[id] = ''
    state.pending[id] = [...(state.pending[id] ?? []), message]
    emitChange()
  })

  on<{ sessionId?: string; call?: ToolCall } | ToolCall>('agent:tool-call', (payload) => {
    const wrapped = payload as { sessionId?: string; call?: ToolCall }
    const call = (wrapped.call ?? payload) as ToolCall
    const id = wrapped.sessionId || activeId
    state.toolCalls[id] = [...(state.toolCalls[id] ?? []), { call }]
    emitChange()
  })

  on<{ sessionId?: string; result?: ToolResult } | ToolResult>('agent:tool-result', (payload) => {
    const wrapped = payload as { sessionId?: string; result?: ToolResult }
    const result = (wrapped.result ?? payload) as ToolResult
    const id = wrapped.sessionId || activeId
    const list = state.toolCalls[id] ?? []
    const match = [...list].reverse().find((entry) => entry.call.id === result.callId && !entry.result)
    if (match) match.result = result
    state.toolCalls[id] = [...list]
    emitChange()
  })

  on<{ sessionId?: string; strategy: string }>('agent:compacted', (info) => {
    const id = info.sessionId || activeId
    state.notices[id] = `Context compacted (${info.strategy}) to keep the session going.`
    emitChange()
  })

  on<{ sessionId?: string; reason?: string } | string>('agent:done', (payload) => {
    const id = typeof payload === 'string' ? activeId : payload.sessionId || activeId
    state.running[id] = false
    emitChange()
  })

  on<{ sessionId?: string; message?: string } | string>('agent:error', (payload) => {
    const id = typeof payload === 'string' ? activeId : payload.sessionId || activeId
    state.errors[id] = typeof payload === 'string' ? payload : (payload.message ?? 'Unknown error')
    state.running[id] = false
    emitChange()
  })

  on<PermissionRequest>('agent:permission-request', (req) => {
    state.permissionQueue = [...state.permissionQueue, req]
    emitChange()
  })
}

wire()
