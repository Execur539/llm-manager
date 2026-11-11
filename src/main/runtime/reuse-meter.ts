/**
 * Why a prompt had to be read again.
 *
 * The server skips whatever a new prompt shares with the last one from its first token, and on a
 * model with recurrent layers it can only go back to a checkpoint, so one changed character near
 * the start costs the whole conversation. Every cause of that found so far was something this
 * app sent differently from the turn before, and none of them was visible from outside: the
 * reply simply took a long time to start.
 *
 * So each request is compared with the one before it, and when the server reports that it read a
 * large share of the prompt, the first place the two differ is named. Pure functions of what was
 * sent, so they can be tested without a server.
 */

import type { ToolDefinition } from '@shared/types'
import type { ChatMessage } from './llama'

/** What a request sent, reduced to what decides whether the server can reuse its cache. */
export interface SentShape {
  /** The conversation the request belongs to, when it belongs to one. */
  key?: string
  /** Everything that reaches the template besides the messages: effort, switches, tool names. */
  settings: string
  tools: string
  /** One entry per message; equal entries render to equal text. */
  messages: string[]
}

export interface RereadReport {
  /** Tokens the server had to read. */
  read: number
  /** Tokens it took from its cache. */
  reused: number
  total: number
  /** Why, in words that can be shown to the person waiting. */
  reason: string
  /** Where the two prompts first differ, for the log. */
  detail?: string
  key?: string
}

/** A message as a string that changes exactly when what the template renders for it changes. */
function messageSignature(m: ChatMessage): string {
  const content =
    typeof m.content === 'string'
      ? m.content
      : m.content
          .map((p) => {
            if (p.type === 'text') return p.text ?? ''
            // Media is megabytes of base64; its length and both ends tell two payloads apart.
            const data = p.image_url?.url ?? p.input_audio?.data ?? p.input_video?.data ?? p.input_video?.url ?? ''
            return `[${p.type}:${data.length}:${data.slice(0, 48)}:${data.slice(-48)}]`
          })
          .join('\u0001')
  const calls = m.tool_calls?.map((c) => `${c.function.name}(${c.function.arguments})`).join('\u0001') ?? ''
  return [m.role, content, m.reasoning_content ?? '', calls, m.tool_call_id ?? ''].join('\u0002')
}

export function shapeOf(opts: {
  cacheKey?: string
  messages: ChatMessage[]
  tools?: ToolDefinition[]
  reasoningEffort?: string
  chatTemplateKwargs?: Record<string, unknown>
}): SentShape {
  return {
    key: opts.cacheKey,
    settings: JSON.stringify({ effort: opts.reasoningEffort ?? null, kwargs: opts.chatTemplateKwargs ?? {} }),
    tools: JSON.stringify((opts.tools ?? []).map((t) => [t.name, t.description, t.parameters])),
    messages: opts.messages.map(messageSignature)
  }
}

/** The text either side of the first character two strings disagree on. */
function firstDifference(a: string, b: string): string {
  let i = 0
  while (i < a.length && i < b.length && a[i] === b[i]) i++
  const show = (s: string): string => JSON.stringify(s.slice(Math.max(0, i - 40), i + 60))
  return `at character ${i}: was ${show(a)}, now ${show(b)}`
}

/** Whether `next` only adds to what `prev` sent, so nothing already read has changed. */
export function extendsPrevious(prev: SentShape | null, next: SentShape): boolean {
  if (!prev) return false
  if (prev.settings !== next.settings || prev.tools !== next.tools) return false
  if (prev.messages.length > next.messages.length) return false
  return prev.messages.every((m, i) => m === next.messages[i])
}

/**
 * Why the server could not reuse what it had, judged from the request before this one.
 *
 * Ordered from the start of the prompt to its end, because the earliest difference is the one
 * that decides how much is read again.
 */
export function explainReread(prev: SentShape | null, next: SentShape): { reason: string; detail?: string } {
  if (!prev) return { reason: 'nothing was in memory yet, so the whole conversation was read' }

  if (prev.settings !== next.settings) {
    return {
      reason: 'the effort level or thinking switch changed, which is written at the very start of the prompt',
      detail: `settings ${firstDifference(prev.settings, next.settings)}`
    }
  }
  if (prev.tools !== next.tools) {
    return { reason: 'the list of tools changed', detail: `tools ${firstDifference(prev.tools, next.tools)}` }
  }

  const shared = Math.min(prev.messages.length, next.messages.length)
  for (let i = 0; i < shared; i++) {
    if (prev.messages[i] === next.messages[i]) continue
    const role = next.messages[i].split('\u0002')[0]
    const reason =
      role === 'system'
        ? 'the system prompt changed'
        : prev.key !== next.key
          ? 'a different conversation was in memory'
          : `message ${i + 1} of the conversation (${role}) was sent differently from last time`
    return { reason, detail: `message ${i + 1} ${firstDifference(prev.messages[i], next.messages[i])}` }
  }

  if (prev.messages.length > next.messages.length) {
    return { reason: 'the conversation was cut back to an earlier point' }
  }
  if (prev.key !== next.key) return { reason: 'a different conversation was in memory' }
  // Nothing this app sent differs, so the difference is on the server's side of the exchange.
  return { reason: 'the server no longer held this conversation, or wrote its last reply differently from how it was sent back' }
}

/** Worth telling someone about: thousands of tokens, and a real share of the prompt. */
export function isLargeReread(read: number, total: number): boolean {
  return read >= 2000 && read >= total * 0.3
}
