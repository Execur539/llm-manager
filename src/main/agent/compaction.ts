/**
 * Compaction planning, kept apart from the agent loop so it can be tested on its own.
 *
 * Three decisions live here: how a transcript is cut into summarisation passes, whether a small
 * model's summary has gone wrong, and where a helper model should run. None of them needs a
 * model, Electron or the network.
 */

import type { ModelArchInfo } from '@shared/types'
import { computeBufferBytes, kvCacheBytes, logitsBufferBytes } from '../autofit/engine'

const MB = 1024 * 1024

/** The runtime's token estimate (`estimateTokens`), repeated here so this module stays pure. */
export const approxTokens = (text: string): number => Math.ceil(text.length / 4)

/** What every summarising pass is asked to do, whichever model does it. */
export const SUMMARY_INSTRUCTION =
  'Summarise the following agent transcript. Preserve: the user goal, decisions made, ' +
  'files and paths touched, commands run and their outcomes, and anything still ' +
  'outstanding. Be dense and factual. No preamble.'

/*
 * Word ceilings the model is asked to keep to, so a summary ends where it means to rather than
 * where its token limit cuts it.
 *
 * Measured on a real 47K-token agent session with only a token limit: five of twelve part
 * summaries stopped mid-sentence, and the merged one ended in the middle of a list — which is the
 * text the loaded model would have kept as its memory of everything before it. The merge also came
 * back as the parts stacked under headings rather than one account, hence the wording.
 */
export const CHUNK_SUMMARY_WORDS = 250
export const MERGED_SUMMARY_WORDS = 700
export const CHUNK_INSTRUCTION = `${SUMMARY_INSTRUCTION} This is one part of a longer transcript; stay under ${CHUNK_SUMMARY_WORDS} words.`
export const MERGE_INSTRUCTION =
  `${SUMMARY_INSTRUCTION} The input is summaries of consecutive parts of one session. Combine them ` +
  `into a single account in order, not a list of parts, and stay under ${MERGED_SUMMARY_WORDS} words.`

/** The same passes over a chat conversation, which has goals and answers rather than files and commands. */
export const CHAT_SUMMARY_INSTRUCTION =
  'Summarise the following conversation between a user and an assistant. Preserve: what the user ' +
  'is trying to do, facts and preferences they gave, answers and decisions reached, code or ' +
  'commands that matter, and questions still open. Be dense and factual. No preamble.'
export const CHAT_CHUNK_INSTRUCTION = `${CHAT_SUMMARY_INSTRUCTION} This is one part of a longer conversation; stay under ${CHUNK_SUMMARY_WORDS} words.`
export const CHAT_MERGE_INSTRUCTION =
  `${CHAT_SUMMARY_INSTRUCTION} The input is summaries of consecutive parts of one conversation. Combine ` +
  `them into a single account in order, not a list of parts, and stay under ${MERGED_SUMMARY_WORDS} words.`

export type TranscriptKind = 'agent' | 'chat'

export function instructionsFor(kind: TranscriptKind): { chunk: string; merge: string } {
  return kind === 'chat'
    ? { chunk: CHAT_CHUNK_INSTRUCTION, merge: CHAT_MERGE_INSTRUCTION }
    : { chunk: CHUNK_INSTRUCTION, merge: MERGE_INSTRUCTION }
}

/**
 * Where to cut a conversation for compaction: everything from the returned index on is kept.
 *
 * Decided by token cost rather than by counting messages — four recent messages can be twenty
 * thousand tokens of tool output — and never keeping fewer than `minKeep`, because a conversation
 * cannot carry on from nothing.
 */
export function keepRecent(costs: number[], keepBudget: number, minKeep = 2): number {
  let kept = 0
  let cut = costs.length
  while (cut > 0) {
    const next = kept + costs[cut - 1]
    if (next > keepBudget && costs.length - cut >= minKeep) break
    kept = next
    cut--
  }
  return cut
}

/** What compaction needs from a helper model; the runtime's summariser service provides it. */
export interface CompactionHelper {
  label: string
  contextTokens: number
  complete(system: string, user: string, maxTokens: number, signal?: AbortSignal): Promise<string>
}

export interface TranscriptItem {
  role: string
  text: string
}

export interface ChunkPlan {
  /** The text of each pass, in order. */
  chunks: string[]
  /** Each chunk's share of the whole transcript, summing to 1 — what progress is measured in. */
  weights: number[]
  /** The size the plan aimed for, in tokens. */
  targetTokens: number
}

/**
 * How much of a conversation one helper pass takes.
 *
 * A tenth, so a compaction runs as about ten steps and its progress can be shown as a
 * percentage rather than a spinner. A window too small for a tenth takes smaller pieces — a
 * twentieth where that fits, and whatever fits where it does not. A floor stops a short
 * transcript being cut into slivers too small to summarise usefully.
 */
export const CHUNK_SHARE = 0.1
export const CHUNK_MIN_TOKENS = 600

export function planChunks(items: TranscriptItem[], usableTokens: number, minTokens = CHUNK_MIN_TOKENS): ChunkPlan {
  const lines = items.map((m) => `${m.role}: ${m.text}\n`)
  const total = lines.reduce((a, l) => a + approxTokens(l), 0)
  const usable = Math.max(1, Math.floor(usableTokens))
  const target = Math.max(1, Math.min(usable, Math.max(minTokens, Math.ceil(total * CHUNK_SHARE))))

  const chunks: string[] = []
  let buffer = ''
  let bufferTokens = 0
  const flush = (): void => {
    if (buffer) chunks.push(buffer)
    buffer = ''
    bufferTokens = 0
  }

  for (const line of lines) {
    const cost = approxTokens(line)
    // A message bigger than the whole window is cut into pieces that fit; nothing is dropped.
    if (cost > usable) {
      flush()
      const size = usable * 4
      for (let at = 0; at < line.length; at += size) chunks.push(line.slice(at, at + size))
      continue
    }
    if (bufferTokens > 0 && bufferTokens + cost > target) flush()
    buffer += line
    bufferTokens += cost
  }
  flush()

  const sizes = chunks.map(approxTokens)
  const sum = sizes.reduce((a, b) => a + b, 0) || 1
  return { chunks, weights: sizes.map((s) => s / sum), targetTokens: target }
}

/** Share of the progress bar the chunk passes cover; merging their summaries takes the rest. */
export const CHUNK_PROGRESS_SHARE = 0.9

/** Percent complete once the first `done` chunks of a plan have been summarised. */
export function chunkProgress(weights: number[], done: number): number {
  if (weights.length <= 1) return done >= weights.length ? 100 : 0
  const share = weights.slice(0, done).reduce((a, w) => a + w, 0)
  return Math.min(Math.round(CHUNK_PROGRESS_SHARE * 100), Math.round(share * CHUNK_PROGRESS_SHARE * 100))
}

/**
 * Group chunk summaries into merge passes that each fit the window.
 *
 * Usually one group. When there are enough summaries that joining them would overflow the
 * helper's window, they are merged in rounds instead, so the final pass never truncates.
 */
export function groupForMerge(summaries: string[], usableTokens: number): string[][] {
  const groups: string[][] = []
  let current: string[] = []
  let tokens = 0
  for (const s of summaries) {
    const cost = approxTokens(s) + 8
    if (current.length && tokens + cost > usableTokens) {
      groups.push(current)
      current = []
      tokens = 0
    }
    current.push(s)
    tokens += cost
  }
  if (current.length) groups.push(current)
  return groups
}

/**
 * Whether a summary has gone wrong the way small models do: by repeating itself.
 *
 * Qwen3.5-2B's own model card warns it is more prone than its larger siblings to loops that stop
 * it finishing. A loop does not fail loudly — it returns a confident page of the same sentence,
 * which would then be the only record of everything it replaced. Two signals: a substantial line
 * that keeps recurring, or a long output whose phrases are mostly repeats.
 */
export function looksLikeALoop(text: string): boolean {
  const counts = new Map<string, number>()
  for (const raw of text.split(/\n+/)) {
    const line = raw.trim()
    if (line.length < 20) continue
    const n = (counts.get(line) ?? 0) + 1
    if (n >= 4) return true
    counts.set(line, n)
  }
  const words = text.toLowerCase().split(/\s+/).filter(Boolean)
  const n = 6
  if (words.length >= 150) {
    const grams = new Set<string>()
    for (let i = 0; i + n <= words.length; i++) grams.add(words.slice(i, i + n).join(' '))
    if (grams.size / (words.length - n + 1) < 0.4) return true
  }
  return false
}

// ---------------------------------------------------------------- the helper's size and place

/**
 * A helper's window: room for one chunk and a merge, sized for the token estimate's worst case.
 *
 * On a model whose cache is only a handful of attention layers — Qwen3.5-2B keeps six — the
 * difference between 32K and 64K is a couple of hundred megabytes, which is cheaper than a chunk
 * that overflows the window and has to go to the loaded model instead.
 */
export const HELPER_MAX_CONTEXT = 65536
/**
 * A chunk's reply budget: room for its word ceiling even in dense text.
 *
 * 512 tokens cut off four of twelve part summaries on a real session despite a 250-word ceiling.
 * Paths, identifiers and code run to around three tokens a word, so 160 words filled it.
 */
export const HELPER_CHUNK_OUTPUT_TOKENS = 768
/** The merged summary is what replaces the transcript, so it gets more room. */
export const HELPER_MERGE_OUTPUT_TOKENS = 1536
/**
 * A smaller physical batch than llama.cpp's 512.
 *
 * The logits buffer is batch x vocabulary x 4 bytes, and Qwen3.5's vocabulary is 248,320 — half a
 * gigabyte at 512 for a model whose weights are 1.2 GB. Halving the batch halves that and costs
 * little prompt speed on a model this small.
 */
export const HELPER_UBATCH = 256

export function helperContext(arch: ModelArchInfo | null): number {
  const trained = arch?.contextLength ?? 0
  return trained > 0 ? Math.min(trained, HELPER_MAX_CONTEXT) : HELPER_MAX_CONTEXT
}

/**
 * How far the character-based estimate can fall short of a real tokenizer.
 *
 * Measured on a real agent transcript: a chunk estimated at 5,576 tokens was 17,411 to Qwen3.5's
 * tokenizer — 3.1 times as many. Paths, identifiers, hashes and JSON break into far more tokens
 * than prose does, and agent transcripts are full of them. Chunks are sized against the window as
 * though the estimate were further out than that, because one that overflowed would fail outright
 * — and at exactly 3.1x a chunk at the limit would still have missed by a percent.
 */
export const TOKEN_ESTIMATE_SAFETY = 3.5

/** Estimated tokens a chunk (or a merge input) may take, once the instruction and the reply are allowed for. */
export function usableChunkTokens(contextTokens: number, instructionTokens: number): number {
  const room = contextTokens - instructionTokens - HELPER_MERGE_OUTPUT_TOKENS - 256
  return Math.max(256, Math.floor(room / TOKEN_ESTIMATE_SAFETY))
}

/** VRAM a helper needs: weights, a cache for its whole window, compute, logits, and a margin. */
export function helperVramNeed(arch: ModelArchInfo, fileBytes: number, contextTokens: number): number {
  return (
    fileBytes +
    kvCacheBytes(arch, contextTokens, 'q8_0') +
    computeBufferBytes(arch, contextTokens, HELPER_UBATCH, true) +
    logitsBufferBytes(arch, HELPER_UBATCH) +
    512 * MB
  )
}

export interface HelperGpu {
  /** CUDA index. The app pins CUDA's numbering to nvidia-smi's, so this is nvidia-smi's index. */
  index: number
  name: string
  freeBytes: number
}

export type HelperPlacement = { kind: 'gpu'; index: number; name: string } | { kind: 'cpu'; reason: string }

/**
 * Where the helper runs: the GPU with the most free memory, if any has room for it, else the CPU.
 *
 * "GPU" as a preference still falls back to the CPU rather than failing, because the free memory
 * belongs to the loaded model first — a helper that pushed the main model out of memory would
 * cost far more time than it saves.
 */
export function chooseHelperPlacement(
  need: number,
  gpus: HelperGpu[],
  preference: 'auto' | 'gpu' | 'cpu'
): HelperPlacement {
  if (preference === 'cpu') return { kind: 'cpu', reason: 'set to run on the CPU' }
  const roomiest = [...gpus].sort((a, b) => b.freeBytes - a.freeBytes).find((g) => g.freeBytes >= need)
  if (roomiest) return { kind: 'gpu', index: roomiest.index, name: roomiest.name }
  return {
    kind: 'cpu',
    reason: gpus.length ? 'no GPU has enough free memory beside the loaded model' : 'no usable GPU'
  }
}
