/**
 * The parts of an agent conversation's prompt that must never change once sent.
 *
 * The server reuses what it has already read only as far as a new prompt matches the last one from
 * its first token. The system prompt and the tool list are that start, and they used to be built
 * fresh every turn with the date, the working folder and the remembered notes inside them: the
 * first message after midnight, a note saved in another session, or a tool server reconnecting
 * each made every open conversation read itself again from the top.
 *
 * So a conversation records the system prompt and tool list it first sent and sends those for as
 * long as it lives. Everything that can change is told to the model after them, at the top of the
 * user's next message, and again only when it changes. The record of what has been told lives
 * here too, so a rebuilt history says the same thing in the same place.
 *
 * Pure, so it can be tested without a model or a database.
 */

import type { ToolDefinition } from '@shared/types'

/** What the model is told about the session it is working in. Anything here may change. */
export interface SessionFacts {
  cwd: string
  scratch: string
  /** Whether commands already run with administrator rights. */
  admin: boolean
  /** YYYY-MM-DD. */
  date: string
  memories: string[]
  planMode: boolean
}

export interface PromptState {
  /** The system prompt this conversation first sent, sent unchanged ever after. */
  system: string
  /** The tools it first offered, in the order it offered them. */
  tools: ToolDefinition[]
  /** Which message told the model what, oldest first. */
  announced: { messageId: string; facts: SessionFacts }[]
  /** What the model had been told when the conversation was last summarised. */
  summaryFacts?: SessionFacts
}

const OPEN = '[Session information from the app, not written by the user]'
const CLOSE = '[End of session information]'
const UPDATE_OPEN = '[Update from the app, not written by the user]'
const UPDATE_CLOSE = '[End of update]'

const PLAN_ON =
  'Plan mode is ON. Only read-class tools (reading, listing, searching, fetching) will run; anything ' +
  'that writes or executes is refused. Investigate, then present a concise written plan for the user to approve.'
const PLAN_OFF = 'Plan mode is now off. Every tool may be used again.'

function adminLine(admin: boolean): string {
  return admin ? 'yes, commands already run elevated' : 'no, use run_elevated for a command that needs them'
}

function memoryLines(memories: string[]): string {
  return memories.map((m) => `- ${m}`).join('\n')
}

/** Everything the model needs to know about the session, for the first message that carries it. */
export function factsBlock(facts: SessionFacts): string {
  const lines = [
    OPEN,
    `Working directory: ${facts.cwd}`,
    `Scratch folder: ${facts.scratch}`,
    'Platform: Windows (Windows PowerShell 5.1)',
    `Administrator rights: ${adminLine(facts.admin)}`,
    `Today's date: ${facts.date}`
  ]
  if (facts.memories.length) {
    lines.push('Things you have remembered from previous sessions:', memoryLines(facts.memories))
  }
  if (facts.planMode) lines.push(PLAN_ON)
  lines.push(CLOSE)
  return lines.join('\n')
}

/** What has changed since the model was last told, or null when nothing has. */
export function factsUpdate(told: SessionFacts, now: SessionFacts): string | null {
  const lines: string[] = []
  if (told.date !== now.date) lines.push(`- Today's date is now ${now.date}.`)
  if (told.cwd !== now.cwd) lines.push(`- The working directory is now ${now.cwd}`)
  if (told.scratch !== now.scratch) lines.push(`- The scratch folder is now ${now.scratch}`)
  if (told.admin !== now.admin) lines.push(`- Administrator rights: now ${adminLine(now.admin)}.`)
  if (told.memories.join('\n') !== now.memories.join('\n')) {
    lines.push(
      now.memories.length
        ? `- The things you have remembered are now, in full:\n${memoryLines(now.memories)}`
        : '- The list of things you have remembered is now empty.'
    )
  }
  if (told.planMode !== now.planMode) lines.push(`- ${now.planMode ? PLAN_ON : PLAN_OFF}`)
  return lines.length ? [UPDATE_OPEN, ...lines, UPDATE_CLOSE].join('\n') : null
}

/**
 * What the model currently believes, given which messages are still in the conversation.
 *
 * A message that told it something can be removed by an edit or a rewind, and with it what it
 * said; the last announcement still standing is the truth. Behind a summary, it is what had been
 * told when the summary was written.
 */
export function toldFacts(state: PromptState, liveMessageIds: Set<string>, hasSummary: boolean): SessionFacts | null {
  for (let i = state.announced.length - 1; i >= 0; i--) {
    if (liveMessageIds.has(state.announced[i].messageId)) return state.announced[i].facts
  }
  return hasSummary && state.summaryFacts ? state.summaryFacts : null
}

/**
 * The note a new user message must carry, and the state once it has.
 *
 * The whole block when the model has been told nothing that still stands, the changes when it has,
 * and nothing at all when it is up to date, which is nearly every turn.
 */
export function noteFor(
  state: PromptState,
  now: SessionFacts,
  liveMessageIds: Set<string>,
  hasSummary: boolean,
  messageId: string
): { note: string | null; state: PromptState } {
  const told = toldFacts(state, liveMessageIds, hasSummary)
  const note = told ? factsUpdate(told, now) : factsBlock(now)
  if (!note) return { note: null, state }
  return {
    note,
    state: {
      ...state,
      // Announcements on messages that no longer exist are dropped as the record is rewritten.
      announced: [...state.announced.filter((a) => liveMessageIds.has(a.messageId)), { messageId, facts: now }]
    }
  }
}

/** A user's text with the app's note above it. */
export function withNote(note: string | null, text: string): string {
  return note ? `${note}\n\n${text}` : text
}

/** The state after a summary has replaced the messages up to a boundary. */
export function afterSummary(state: PromptState, told: SessionFacts | null, liveMessageIds: Set<string>): PromptState {
  return {
    ...state,
    announced: state.announced.filter((a) => liveMessageIds.has(a.messageId)),
    summaryFacts: told ?? state.summaryFacts
  }
}

/** A stored state read back, or null when it is missing or not one. */
export function parsePromptState(raw: string | null | undefined): PromptState | null {
  if (!raw) return null
  try {
    const s = JSON.parse(raw) as Partial<PromptState>
    if (typeof s.system !== 'string' || !Array.isArray(s.tools) || !Array.isArray(s.announced)) return null
    return { system: s.system, tools: s.tools, announced: s.announced, summaryFacts: s.summaryFacts }
  } catch {
    return null
  }
}
