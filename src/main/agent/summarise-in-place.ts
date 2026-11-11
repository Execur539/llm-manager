/**
 * Summarising a conversation from inside it.
 *
 * The other way to summarise hands the transcript to the model as text in a new prompt, so the
 * model reads the whole older half of the conversation a second time before it writes a word. But
 * the server is already holding that conversation: asking for the summary as the conversation's
 * own next message costs no reading at all, only the writing.
 *
 * For that to hold, the request has to start exactly as the conversation's last one did: the same
 * tools, the same effort level, the same messages. So the tools are still sent, and calling them
 * is forbidden rather than made impossible; thinking is switched off through the server's budget
 * rather than through the template, which would rewrite the first line of the prompt.
 *
 * Returns null whenever the result cannot be trusted or there is no room to ask, and the caller
 * falls back to the pass-by-pass summary.
 */

import type { ToolDefinition } from '@shared/types'
import { llama, type ChatMessage, type CompletionOptions } from '../runtime/llama'
import { logger } from '../log'
import {
  IN_PLACE_MAX_TOKENS,
  IN_PLACE_WORDS,
  hasRoomInPlace,
  inPlaceInstruction,
  usableSummary,
  type TranscriptKind
} from './compaction'

export async function summariseInPlace(opts: {
  kind: TranscriptKind
  /** The conversation exactly as its last request sent it, plus the reply to that request. */
  messages: ChatMessage[]
  tools?: ToolDefinition[]
  cacheKey?: string
  /** The effort fields the conversation's requests carry; anything else changes the start of the prompt. */
  effort?: Pick<CompletionOptions, 'reasoningEffort' | 'chatTemplateKwargs' | 'reasoningBudget'>
  usedTokens: number
  signal?: AbortSignal
  background?: boolean
  onProgress?: (percent: number) => void
}): Promise<string | null> {
  const loaded = llama.loaded
  if (!loaded || !opts.messages.length) return null
  if (!hasRoomInPlace(opts.usedTokens, loaded.plan.contextLength)) return null

  let text = ''
  let called = false
  try {
    const events = llama.streamEvents({
      ...opts.effort,
      messages: [...opts.messages, { role: 'user', content: inPlaceInstruction(opts.kind) }],
      tools: opts.tools,
      toolChoice: opts.tools?.length ? 'none' : undefined,
      cacheKey: opts.cacheKey,
      // The server's own switch: it closes the thought without the template being told anything.
      reasoningBudget: 0,
      maxTokens: IN_PLACE_MAX_TOKENS,
      temperature: 0.3,
      background: opts.background,
      signal: opts.signal
    })
    for await (const ev of events) {
      if (ev.type === 'text') {
        text += ev.text
        opts.onProgress?.(Math.min(95, Math.round((text.split(/\s+/).length / IN_PLACE_WORDS) * 100)))
      } else if (ev.type === 'tool_call') {
        called = true
      }
    }
  } catch (err) {
    if (!opts.signal?.aborted) logger.warn('compaction', 'summarising in place failed; falling back to passes', err)
    return null
  }
  if (opts.signal?.aborted) return null
  if (called || !usableSummary(text)) {
    logger.warn('compaction', 'the summary written in place was not usable; falling back to passes', {
      called,
      words: text.trim().split(/\s+/).filter(Boolean).length
    })
    return null
  }
  return text.trim()
}
