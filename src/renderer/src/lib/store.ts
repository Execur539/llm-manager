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
import type { AgentMessage, AgentQuestion, PermissionRequest, ToolCall, ToolResult } from '@shared/types'
import { on } from './api'

/** One of Ultra's independent attempts, as it appears in the transcript. */
export interface UltraSampleView {
  index: number
  /** Grows while this sample is the one running. */
  answer: string
  reasoning: string
  /** Set once the sample is final; a sample still streaming has no summary yet. */
  done: boolean
  continuations?: number
  temperature?: number
  ms?: number
}

export interface StreamState {
  /** partial assistant text, keyed by chat/session id */
  partial: Record<string, string>
  /**
   * The model's chain of thought for the turn in flight, keyed the same way.
   *
   * Kept separate from `partial` because it is a different thing being said — the UI shows it in
   * its own collapsible block rather than inline with the answer.
   */
  reasoningPartial: Record<string, string>
  /**
   * Prompt-reading progress for the turn in flight, keyed the same way, or absent once
   * generation has started.
   *
   * This covers the gap between sending and the first token, which for a long conversation or
   * a freshly loaded model is the part of a turn that feels broken: nothing moves, and there is
   * no way to tell a slow prompt from a hung one.
   */
  promptProgress: Record<string, { percent: number; processed: number; total: number; cached: number }>
  /**
   * How much of the model's context window a conversation is occupying, and how big it is.
   *
   * Unlike everything else here this is not per-turn state and is deliberately not cleared when
   * one starts or ends — the tokens stay in the window between turns, so a reading that vanished
   * the moment a response finished would be describing something that had not changed. It goes
   * when the conversation does.
   */
  context: Record<string, { used: number; max: number }>
  /**
   * A compaction in progress, and when it began.
   *
   * Auto-compaction is a whole model call over the older half of a session, which on a long one
   * runs for many seconds. It used to happen in complete silence between iterations of a turn,
   * so the app appeared to have stopped, and a message sent during it would be appended to a
   * history that was about to be rewritten underneath it. The start time is here rather than the
   * elapsed seconds because a ticking number belongs to whatever is rendering it, not to state
   * every subscriber re-renders for.
   */
  compacting: Record<string, { since: number; strategy: string; automatic: boolean }>
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
  /** clarifying questions the agent is blocked on; same reasoning as the permission queue */
  questionQueue: AgentQuestion[]
  /** tool calls seen this turn, so a remount can still render them in order */
  toolCalls: Record<string, { call: ToolCall; result?: ToolResult }[]>
  /**
   * Tool calls the model is still writing, keyed by conversation and ordered by call index.
   *
   * A call cannot be dispatched until its arguments have finished arriving, so between the model
   * deciding to act and the call being ready there was nothing on screen at all. `args` is the
   * raw, usually incomplete JSON accumulated so far — for showing, not for parsing.
   */
  pendingCalls: Record<string, { index: number; name: string; args: string }[]>
  /**
   * Ultra's independent attempts for the turn in flight, keyed by conversation.
   *
   * Held apart from `partial` on purpose: these are drafts, and the place the real answer will
   * appear must not fill up with text that is about to be replaced. They render as their own
   * collapsed boxes above the answer, and clear when the turn ends.
   */
  ultra: Record<string, UltraSampleView[]>
  /** true once sampling is done and the synthesised answer is being written */
  ultraSynthesising: Record<string, boolean>
  /** The plan Ultra settled on for an agent turn, shown above the work it directs. */
  ultraPlan: Record<string, string>
  /**
   * Which conversation each view has open.
   *
   * This has to outlive the view for the same reason the streamed text does: navigating away
   * unmounts the view, and if the selection lived in component state the user would come back
   * to nothing selected — losing sight of a response that is still arriving.
   */
  selection: { chat: string | null; agent: string | null }
  /**
   * Transient confirmations, newest last.
   *
   * Several actions used to complete in total silence — exporting a conversation opened a file
   * dialog and then gave no sign whether anything had been written, which is indistinguishable
   * from a broken button. Anything the user cannot otherwise see the result of reports here.
   */
  toasts: Toast[]
  /**
   * Whether the conversation rail is showing as an overlay.
   *
   * Only meaningful on narrow viewports — the remote UI on a phone — where the rail is a drawer
   * rather than a permanent column. It lives here because the control that opens it sits in the
   * view header while the thing it opens is a sibling component.
   */
  railOpen: boolean
  /**
   * Reasoning effort per conversation, as named by the loaded model's own template.
   * An absent entry means "leave the template's default alone".
   */
  reasoning: Record<string, string>
}

export interface Toast {
  id: string
  message: string
  kind: 'info' | 'success' | 'error'
  /** optional absolute path the toast can offer to reveal in the file manager */
  revealPath?: string
}

/**
 * Where the open-conversation selection is remembered across reloads.
 *
 * Everything else in this store is deliberately in-memory — it describes a turn in flight. The
 * selection is different: it is a place the user was, and losing it on every restart meant
 * reopening the app landed on an empty pane with the conversation sitting unselected in the rail
 * beside it.
 */
const SELECTION_KEY = 'llmm.selection'

function loadSelection(): { chat: string | null; agent: string | null } {
  try {
    const raw = JSON.parse(localStorage.getItem(SELECTION_KEY) ?? 'null')
    return {
      chat: typeof raw?.chat === 'string' ? raw.chat : null,
      agent: typeof raw?.agent === 'string' ? raw.agent : null
    }
  } catch {
    return { chat: null, agent: null }
  }
}

function saveSelection(selection: { chat: string | null; agent: string | null }): void {
  try {
    localStorage.setItem(SELECTION_KEY, JSON.stringify(selection))
  } catch {
    // Private browsing, a full quota — not worth failing a click over.
  }
}

const state: StreamState = {
  partial: {},
  promptProgress: {},
  context: {},
  compacting: {},
  reasoningPartial: {},
  running: {},
  pending: {},
  errors: {},
  notices: {},
  permissionQueue: [],
  questionQueue: [],
  toolCalls: {},
  pendingCalls: {},
  ultra: {},
  ultraSynthesising: {},
  ultraPlan: {},
  selection: loadSelection(),
  toasts: [],
  railOpen: false,
  reasoning: {}
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
    promptProgress: { ...state.promptProgress },
    context: { ...state.context },
    compacting: { ...state.compacting },
    reasoningPartial: { ...state.reasoningPartial },
    running: { ...state.running },
    pending: { ...state.pending },
    errors: { ...state.errors },
    notices: { ...state.notices },
    permissionQueue: [...state.permissionQueue],
    questionQueue: [...state.questionQueue],
    toolCalls: { ...state.toolCalls },
    pendingCalls: { ...state.pendingCalls },
    ultra: { ...state.ultra },
    ultraSynthesising: { ...state.ultraSynthesising },
    ultraPlan: { ...state.ultraPlan },
    selection: { ...state.selection },
    toasts: [...state.toasts],
    railOpen: state.railOpen,
    reasoning: { ...state.reasoning }
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
    state.pendingCalls[id] = []
    state.ultra[id] = []
    state.ultraSynthesising[id] = false
    delete state.ultraPlan[id]
    // Last turn's figure would otherwise flash before the first frame of this one arrives.
    delete state.promptProgress[id]
  } else {
    // The samples were scaffolding for an answer that now exists on its own.
    delete state.ultra[id]
    delete state.ultraSynthesising[id]
    delete state.ultraPlan[id]
    // A turn that ended without ever generating — stopped, or failed — must not leave a
    // progress bar sitting at whatever it had reached.
    delete state.promptProgress[id]
    /*
     * Nor a tool call that is forever about to be written.
     *
     * A stopped turn abandons whatever call was mid-composition, and nothing else clears it —
     * the finished `tool_call` that normally supersedes it is exactly what never arrives. Left
     * alone it sat there claiming the agent was still writing, with nothing running at all.
     */
    delete state.pendingCalls[id]
  }
  emitChange()
}

/**
 * Restore a conversation's context reading from what was stored with it.
 *
 * Only fills a gap: a live figure from a turn in flight is always closer to the truth than a
 * number written at the end of the last one, so it is never overwritten. The stored tokens were
 * counted by whichever model was loaded then, while the maximum comes from the model loaded now
 * — reopening under a different model is approximate until its first turn reports for itself.
 */
export function seedContext(id: string, used: number | undefined, max: number): void {
  if (!id || !used || !max || state.context[id]) return
  state.context[id] = { used, max }
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

export function clearQuestions(): void {
  if (!state.questionQueue.length) return
  state.questionQueue = []
  emitChange()
}

export function clearFor(id: string): void {
  if (state.selection.chat === id) state.selection.chat = null
  if (state.selection.agent === id) state.selection.agent = null
  saveSelection(state.selection)
  delete state.partial[id]
  delete state.promptProgress[id]
  delete state.reasoningPartial[id]
  delete state.running[id]
  delete state.pending[id]
  delete state.errors[id]
  delete state.notices[id]
  delete state.toolCalls[id]
  delete state.pendingCalls[id]
  // The conversation is gone, so its context reading and any compaction go with it. These are
  // the only place they are removed: both deliberately outlive individual turns.
  delete state.context[id]
  delete state.compacting[id]
  delete state.ultra[id]
  delete state.ultraSynthesising[id]
  delete state.ultraPlan[id]
  delete state.reasoning[id]
  emitChange()
}

export function dismissPermission(requestId: string): void {
  state.permissionQueue = state.permissionQueue.filter((r) => r.id !== requestId)
  emitChange()
}

export function dismissQuestion(questionId: string): void {
  state.questionQueue = state.questionQueue.filter((q) => q.id !== questionId)
  emitChange()
}

export function setReasoning(id: string, choice: string | null): void {
  if (choice === null) delete state.reasoning[id]
  else state.reasoning[id] = choice
  emitChange()
}

/**
 * Where the effort choice lives before there is a conversation to attach it to.
 *
 * The control sits in the composer, which is usable from the moment the view opens — but the
 * choice is keyed by conversation, and a conversation does not exist until the first message is
 * sent. Without somewhere to put it, the setting was written against `null` and dropped: in the
 * agent view, which never restores a session on mount, that made the slider appear broken, and
 * in chat it silently discarded whatever was chosen before the opening message.
 */
export const DRAFT_CHAT = 'draft:chat'
export const DRAFT_AGENT = 'draft:agent'

/** Carry the draft choice onto the conversation that has just been created for it. */
export function adoptReasoning(fromId: string, toId: string): void {
  const choice = state.reasoning[fromId]
  if (choice === undefined) return
  state.reasoning[toId] = choice
  emitChange()
}

export function toggleRail(): void {
  state.railOpen = !state.railOpen
  emitChange()
}

export function closeRail(): void {
  if (!state.railOpen) return
  state.railOpen = false
  emitChange()
}

let toastSeq = 0

/**
 * Show a transient confirmation. Errors stay until dismissed; successes clear themselves,
 * because a message the user has already read should not need a click to get rid of.
 */
export function toast(message: string, kind: Toast['kind'] = 'info', revealPath?: string): string {
  const id = `t${++toastSeq}`
  state.toasts = [...state.toasts, { id, message, kind, revealPath }]
  emitChange()
  if (kind !== 'error') {
    setTimeout(() => dismissToast(id), 6000)
  }
  return id
}

export function dismissToast(id: string): void {
  const next = state.toasts.filter((t) => t.id !== id)
  if (next.length === state.toasts.length) return
  state.toasts = next
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
  saveSelection(state.selection)
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
  /*
   * Prompt progress, and the two events that end it.
   *
   * The server does send a closing frame at 100%, so the bar is never left stranded partway —
   * but it is the first token that actually retires it, because between that frame and the
   * first token there is still a pause worth showing something for. Content and reasoning both
   * count: a reasoning model's first visible act is a thought, not an answer.
   */
  on<{ chatId: string; percent: number; processed: number; total: number; cached: number }>(
    'chat:prompt-progress',
    (d) => {
      const id = d.chatId || activeId
      state.promptProgress[id] = {
        percent: d.percent,
        processed: d.processed,
        total: d.total,
        cached: d.cached
      }
      emitChange()
    }
  )

  on<{ chatId?: string; used: number; max: number }>('chat:context', (d) => {
    const id = d.chatId || activeId
    state.context[id] = { used: d.used, max: d.max }
    emitChange()
  })

  on<{ sessionId?: string; used: number; max: number }>('agent:context', (d) => {
    const id = d.sessionId || activeId
    state.context[id] = { used: d.used, max: d.max }
    emitChange()
  })

  on<{ chatId: string; text: string }>('chat:delta', (d) => {
    const id = d.chatId || activeId
    state.partial[id] = (state.partial[id] ?? '') + d.text
    delete state.promptProgress[id]
    emitChange()
  })

  on<{ chatId: string; text: string }>('chat:reasoning', (d) => {
    const id = d.chatId || activeId
    state.reasoningPartial[id] = (state.reasoningPartial[id] ?? '') + d.text
    delete state.promptProgress[id]
    emitChange()
  })

  on<{ chatId: string; message: AgentMessage }>('chat:message', (d) => {
    const id = d.chatId || activeId
    state.partial[id] = ''
    delete state.promptProgress[id]
    // The finished message carries its own reasoning; the streamed copy has served its purpose.
    state.reasoningPartial[id] = ''
    state.pending[id] = [...(state.pending[id] ?? []), d.message]
    emitChange()
  })

  // ---- Ultra
  //
  // Samples accumulate against the conversation and are wiped when the turn's real answer
  // arrives: they are working, not transcript, and persisting them would mean every reopened
  // conversation carried three discarded drafts above each answer.

  const sampleAt = (id: string, index: number): UltraSampleView => {
    const list = (state.ultra[id] ??= [])
    let sample = list.find((s) => s.index === index)
    if (!sample) {
      sample = { index, answer: '', reasoning: '', done: false }
      list.push(sample)
      list.sort((a, b) => a.index - b.index)
    }
    return sample
  }

  on<{ chatId: string; index: number; total: number }>('chat:ultra-sample-start', (d) => {
    const id = d.chatId || activeId
    // A fresh run starts from nothing, so a retry does not stack onto the previous attempt.
    if (d.index === 0) state.ultra[id] = []
    state.ultraSynthesising[id] = false
    sampleAt(id, d.index)
    emitChange()
  })

  on<{ chatId: string; index: number; text: string }>('chat:ultra-sample-delta', (d) => {
    sampleAt(d.chatId || activeId, d.index).answer += d.text
    emitChange()
  })

  on<{ chatId: string; index: number; text: string }>('chat:ultra-sample-reasoning', (d) => {
    sampleAt(d.chatId || activeId, d.index).reasoning += d.text
    emitChange()
  })

  on<{
    chatId: string
    sample: { index: number; answer: string; reasoning: string; continuations: number; temperature: number; ms: number }
  }>('chat:ultra-sample', (d) => {
    const id = d.chatId || activeId
    const sample = sampleAt(id, d.sample.index)
    // Take the finished text over the accumulated stream: a forced continuation can replace the
    // answer wholesale, and the deltas alone would show both attempts run together.
    sample.answer = d.sample.answer
    sample.reasoning = d.sample.reasoning
    sample.continuations = d.sample.continuations
    sample.temperature = d.sample.temperature
    sample.ms = d.sample.ms
    sample.done = true
    emitChange()
  })

  on<{ chatId: string }>('chat:ultra-synthesis', (d) => {
    state.ultraSynthesising[d.chatId || activeId] = true
    emitChange()
  })

  /*
   * The agent emits the same shapes under its own channels, keyed by session instead of chat.
   * Same state, same rendering — an Ultra attempt looks the same whether it was an answer or a
   * plan, and the store has no reason to tell them apart.
   */
  on<{ sessionId: string; index: number; total: number }>('agent:ultra-sample-start', (d) => {
    const id = d.sessionId || activeId
    if (d.index === 0) state.ultra[id] = []
    state.ultraSynthesising[id] = false
    sampleAt(id, d.index)
    emitChange()
  })

  on<{ sessionId: string; index: number; text: string }>('agent:ultra-sample-delta', (d) => {
    sampleAt(d.sessionId || activeId, d.index).answer += d.text
    emitChange()
  })

  on<{ sessionId: string; index: number; text: string }>('agent:ultra-sample-reasoning', (d) => {
    sampleAt(d.sessionId || activeId, d.index).reasoning += d.text
    emitChange()
  })

  on<{
    sessionId: string
    sample: { index: number; answer: string; reasoning: string; continuations: number; temperature: number; ms: number }
  }>('agent:ultra-sample', (d) => {
    const sample = sampleAt(d.sessionId || activeId, d.sample.index)
    sample.answer = d.sample.answer
    sample.reasoning = d.sample.reasoning || sample.reasoning
    sample.continuations = d.sample.continuations
    sample.temperature = d.sample.temperature
    sample.ms = d.sample.ms
    sample.done = true
    emitChange()
  })

  on<{ sessionId: string }>('agent:ultra-synthesis', (d) => {
    state.ultraSynthesising[d.sessionId || activeId] = true
    emitChange()
  })

  on<{ sessionId: string; plan: string }>('agent:ultra-plan', (d) => {
    state.ultraPlan[d.sessionId || activeId] = d.plan
    emitChange()
  })

  on<{ sessionId?: string; percent: number; processed: number; total: number; cached: number }>(
    'agent:prompt-progress',
    (d) => {
      const id = d.sessionId || activeId
      state.promptProgress[id] = {
        percent: d.percent,
        processed: d.processed,
        total: d.total,
        cached: d.cached
      }
      emitChange()
    }
  )

  on<{ sessionId?: string; text: string } | string>('agent:delta', (payload) => {
    const id = typeof payload === 'string' ? activeId : payload.sessionId || activeId
    const text = typeof payload === 'string' ? payload : payload.text
    state.partial[id] = (state.partial[id] ?? '') + text
    delete state.promptProgress[id]
    emitChange()
  })

  on<{ sessionId?: string; text: string } | string>('agent:reasoning', (payload) => {
    const id = typeof payload === 'string' ? activeId : payload.sessionId || activeId
    const text = typeof payload === 'string' ? payload : payload.text
    state.reasoningPartial[id] = (state.reasoningPartial[id] ?? '') + text
    delete state.promptProgress[id]
    emitChange()
  })

  on<{ sessionId?: string; message?: AgentMessage } | AgentMessage>('agent:message', (payload) => {
    const wrapped = payload as { sessionId?: string; message?: AgentMessage }
    const message = (wrapped.message ?? payload) as AgentMessage
    const id = wrapped.sessionId || activeId
    state.partial[id] = ''
    state.reasoningPartial[id] = ''
    /*
     * Replace, rather than append, when the id is already queued.
     *
     * The user's turn is announced before Ultra starts planning and stored again when the loop
     * finally runs, so the same message arrives twice by design. The later copy is the complete
     * one — it carries the chosen plan — and the view only dedupes against what it has already
     * drained, so two undrained copies would both be shown.
     */
    state.pending[id] = [...(state.pending[id] ?? []).filter((m) => m.id !== message.id), message]
    // Once the plan is on the message it belongs to, the live copy is a duplicate of it — both
    // were rendering at once for the rest of the turn.
    if (message.plan) delete state.ultraPlan[id]
    emitChange()
  })

  on<{ sessionId?: string; index: number; name: string; args: string }>(
    'agent:tool-call-partial',
    (d) => {
      const id = d.sessionId || activeId
      const list = state.pendingCalls[id] ?? []
      const at = list.findIndex((c) => c.index === d.index)
      const next = { index: d.index, name: d.name, args: d.args }
      state.pendingCalls[id] = at === -1 ? [...list, next] : list.map((c, i) => (i === at ? next : c))
      emitChange()
    }
  )

  on<{ sessionId?: string; call?: ToolCall } | ToolCall>('agent:tool-call', (payload) => {
    const wrapped = payload as { sessionId?: string; call?: ToolCall }
    const call = (wrapped.call ?? payload) as ToolCall
    const id = wrapped.sessionId || activeId
    state.toolCalls[id] = [...(state.toolCalls[id] ?? []), { call }]
    /*
     * The finished calls supersede everything that was being written.
     *
     * They all arrive together, once the stream ends, so the first real call means every
     * in-progress one is now complete and about to be rendered properly.
     */
    state.pendingCalls[id] = []
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

  on<{ sessionId?: string; strategy: string; automatic?: boolean }>('agent:compacting', (info) => {
    const id = info.sessionId || activeId
    state.compacting[id] = {
      since: Date.now(),
      strategy: info.strategy,
      automatic: info.automatic !== false
    }
    emitChange()
  })

  on<{ sessionId?: string; strategy: string }>('agent:compacted', (info) => {
    const id = info.sessionId || activeId
    // Whatever happened, the compaction is over — this is what releases the composer.
    delete state.compacting[id]
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

  on<AgentQuestion>('agent:question', (q) => {
    state.questionQueue = [...state.questionQueue, q]
    emitChange()
  })
}

wire()
