/**
 * The recall tools as the agent has them: looking back into the session's own transcript, for what
 * a compaction left out of what the model is sent. See chat/recall.ts for how they search.
 */

import type { AgentMessage } from '@shared/types'
import type { Tool } from './base'
import { RECALL_TOOLS, recallRead, recallSearch, type RecallOptions } from '../../chat/recall'

export interface RecallSource extends RecallOptions {
  /** The session's messages as they are now, a turn in progress included. */
  messages: AgentMessage[]
}

let source: ((sessionId: string) => RecallSource | null) | null = null

/** Set once by the app: where a session's transcript comes from. */
export function setRecallSource(fn: (sessionId: string) => RecallSource | null): void {
  source = fn
}

const [search, read] = RECALL_TOOLS

export const recallTools: Tool[] = [
  {
    ...search,
    async run(args, ctx) {
      const from = source?.(ctx.sessionId)
      if (!from) return 'The earlier conversation cannot be read right now.'
      return recallSearch(from.messages, args, { ...from, signal: ctx.signal })
    }
  },
  {
    ...read,
    async run(args, ctx) {
      const from = source?.(ctx.sessionId)
      if (!from) return 'The earlier conversation cannot be read right now.'
      return recallRead(from.messages, args, from)
    }
  }
]
