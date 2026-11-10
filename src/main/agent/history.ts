/**
 * The parts of the agent loop that are plain functions of a transcript.
 *
 * Kept apart from the loop so they can be tested without a model, a window or a tool registry.
 */

import type { AgentMessage } from '@shared/types'
import type { ChatMessage, ContentPart } from '../runtime/llama'

export interface RebuildOptions {
  preserveReasoning: boolean
  /** What a stored message sent beyond its text: attachments on a user turn, images on a tool result. */
  sentContentFor?: (messageId: string) => string | ContentPart[] | undefined
}

/** A stored transcript, as the messages the model is sent for it. */
export function rebuildHistory(rest: AgentMessage[], opts: RebuildOptions): ChatMessage[] {
  /*
   * Tool use goes back as tool use.
   *
   * This used to rebuild an assistant turn as its words alone and every tool result as a message
   * from the user, so a history rebuilt from storage — which is every turn — showed the model
   * saying "let me check that" and stopping, over and over, with the user then pasting in what
   * it had wanted. It learned from its own transcript to announce a step and end the turn. The
   * rebuilt prompt also differed from the one the previous turn had sent, from that turn's first
   * call onwards, so none of it could be served from the prompt cache.
   *
   * A result is stored with the call that produced it, which is all that is needed to put the
   * calls back on the assistant turn that made them. Results of one step share a `batch`; stored
   * sessions from before that existed are grouped by adjacency instead.
   */
  const rebuilt: ChatMessage[] = []
  /** The assistant entry the current run of tool results belongs to, and that run's batch. */
  let caller: ChatMessage | null = null
  let callerBatch: string | undefined

  for (const m of rest) {
    if (m.role === 'tool') {
      const call = m.toolCalls?.[0]
      if (!call) {
        // Nothing recorded to rebuild a call from; the old shape is the best that can be done.
        caller = null
        rebuilt.push({ role: 'user', content: `[tool result]\n${m.content}` })
        continue
      }
      if (!caller || (call.batch !== undefined && callerBatch !== undefined && call.batch !== callerBatch)) {
        // A step that called without a word or a thought stored no message of its own.
        caller = { role: 'assistant', content: '', tool_calls: [] }
        rebuilt.push(caller)
      }
      callerBatch = call.batch
      caller.tool_calls = [
        ...(caller.tool_calls ?? []),
        { id: call.id, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.args ?? {}) } }
      ]
      rebuilt.push({ role: 'tool', content: m.content, tool_call_id: call.id, name: call.name })
      // An image the tool returned follows its result as its own turn, as the model first saw it.
      const returned = opts.sentContentFor?.(m.id)
      if (Array.isArray(returned) && returned.length) rebuilt.push({ role: 'user', content: returned })
      continue
    }

    /*
     * A user message goes back exactly as it was sent: attached documents' text, images, audio,
     * video and any Ultra plan included. The transcript stores only what the user typed, so a
     * rebuilt history — Continue, an edit, every agent turn, a restart — used to send all of it
     * as nothing, and the model answered as if it had never seen it.
     */
    const sent = m.role === 'user' ? opts.sentContentFor?.(m.id) : undefined
    if (sent !== undefined) {
      caller = null
      rebuilt.push({ role: 'user', content: sent })
      continue
    }
    /*
     * Past reasoning goes back with the answer it produced, so the model can remember why it
     * decided what it did — not only what it said. `--reasoning-preserve` is what stops the
     * template from dropping all but the latest.
     */
    const entry: ChatMessage = {
      role: m.role as 'user' | 'assistant',
      content: m.content,
      ...(opts.preserveReasoning && m.role === 'assistant' && m.reasoning
        ? { reasoning_content: m.reasoning }
        : {})
    }
    rebuilt.push(entry)
    // Tool results directly after an assistant message are the calls that message made.
    caller = m.role === 'assistant' ? entry : null
    callerBatch = undefined
  }
  return rebuilt
}

/**
 * Whether a reply ends by saying what it is about to do, rather than by finishing or asking.
 *
 * Judged on the last sentence of what the user can read, or of the thinking when nothing was said.
 * A closing question is the model handing the turn back on purpose, and "let me know" is a
 * sign-off, so neither counts.
 */
export function announcesNextStep(content: string, reasoning: string): boolean {
  const source = content.trim() || reasoning.trim()
  if (!source) return false
  const lastLine =
    source
      .split(/\n+/)
      .filter((l) => l.trim())
      .at(-1) ?? ''
  const sentences = lastLine.split(/(?<=[.!?:])\s+/).filter((x) => x.trim())
  const last = (sentences.at(-1) ?? '').trim()
  if (!last || last.endsWith('?')) return false
  if (/\blet me know\b/i.test(last)) return false
  return /^(?:(?:now|next|first|then|so|ok(?:ay)?|alright)[,:]?\s+)?(?:let me|let's|i'll|i will|i'm going to|i am going to|i need to (?:check|run|look|read|verify)|time to)\b/i.test(
    last
  )
}
