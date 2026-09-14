/**
 * Summarising a transcript in passes, for any conversation that has outgrown its window.
 *
 * Shared by the agent and by chat, so both get the same chunking, progress reporting, helper model
 * and fallbacks. The loaded model does whatever the helper cannot.
 */

import { llama, estimateTokens } from '../runtime/llama'
import {
  chunkProgress,
  groupForMerge,
  instructionsFor,
  looksLikeALoop,
  planChunks,
  usableChunkTokens,
  HELPER_CHUNK_OUTPUT_TOKENS,
  HELPER_MERGE_OUTPUT_TOKENS,
  type CompactionHelper,
  type TranscriptItem,
  type TranscriptKind
} from './compaction'

export interface SummaryProgress {
  percent: number
  chunk: number
  chunks: number
  stage: 'chunk' | 'merge'
  /** The helper model doing the work, or null when the loaded model is. */
  helper: string | null
}

/**
 * How much the loaded model reads per pass when there is no helper: about 24,000 characters, the
 * size manual compaction has always used. Kept large, because each extra pass costs a reply
 * written at that model's speed.
 */
const MAIN_PASS_TOKENS = 6000

/** What a summary says when no pass produced anything, so callers can tell it from a real one. */
export const COMPACTION_FAILED = '(compaction failed; earlier turns are described only by this note)'

/**
 * Summarise a run of messages in passes, reporting progress as each one finishes.
 *
 * With a helper model the transcript is always cut into about ten pieces — a tenth each where the
 * helper's window allows — so the interface can say how far through it is, and any piece the
 * helper fails or loops on is done by the loaded model instead. Without one, the loaded model
 * takes bigger pieces: every extra pass costs it a reply written at its own speed, which on a
 * large model is most of the time a compaction takes.
 */
export async function summariseTranscript(
  items: TranscriptItem[],
  opts: {
    kind: TranscriptKind
    helper: CompactionHelper | null
    signal?: AbortSignal
    onProgress?: (p: SummaryProgress) => void
  }
): Promise<string> {
  const { helper, signal } = opts
  const instructions = instructionsFor(opts.kind)

  const byMain = (instruction: string, text: string, maxTokens: number): Promise<string> =>
    llama
      .complete({
        messages: [
          { role: 'system', content: instruction },
          { role: 'user', content: text }
        ],
        temperature: 0.2,
        maxTokens,
        signal
      })
      .then((s) => s.trim())
      .catch(() => '')
  const byHelper = async (instruction: string, text: string, maxTokens: number): Promise<string> => {
    if (!helper) return ''
    const out = await helper.complete(instruction, text, maxTokens, signal).catch(() => '')
    // A loop reads as a confident summary, and would become the only record of what it replaced.
    return out && !looksLikeALoop(out) ? out : ''
  }

  // Sized against the longer of the two instructions, so a merge input fits wherever a chunk did.
  const usable = helper ? usableChunkTokens(helper.contextTokens, estimateTokens(instructions.merge)) : MAIN_PASS_TOKENS
  const plan = helper ? planChunks(items, usable) : planChunks(items, MAIN_PASS_TOKENS, MAIN_PASS_TOKENS)
  const progress = (done: number, stage: 'chunk' | 'merge', percent?: number): void => {
    opts.onProgress?.({
      percent: percent ?? chunkProgress(plan.weights, done),
      chunk: Math.min(done + 1, plan.chunks.length),
      chunks: plan.chunks.length,
      stage,
      helper: helper?.label ?? null
    })
  }

  const parts: string[] = []
  for (let i = 0; i < plan.chunks.length && !signal?.aborted; i++) {
    progress(i, 'chunk')
    const out =
      (await byHelper(instructions.chunk, plan.chunks[i], HELPER_CHUNK_OUTPUT_TOKENS)) ||
      (await byMain(instructions.chunk, plan.chunks[i], helper ? HELPER_CHUNK_OUTPUT_TOKENS : 1024))
    if (out) parts.push(out)
  }

  if (!parts.length) return COMPACTION_FAILED

  // One account rather than several, merged in rounds when the parts would not fit one pass.
  let layer = parts
  while (layer.length > 1 && !signal?.aborted) {
    progress(plan.chunks.length, 'merge')
    const groups = helper ? groupForMerge(layer, usable) : [layer]
    const next: string[] = []
    for (const group of groups) {
      const joined = group.map((p, i) => `Part ${i + 1}:\n${p}`).join('\n\n')
      const merged =
        (await byHelper(instructions.merge, joined, HELPER_MERGE_OUTPUT_TOKENS)) ||
        (await byMain(instructions.merge, joined, 1024))
      next.push(merged || group.join('\n\n'))
    }
    // A round that could not reduce the count would go round forever; keep what it produced.
    layer = next.length < layer.length ? next : [next.join('\n\n')]
  }
  progress(plan.chunks.length, 'merge', 100)
  return layer[0]
}
