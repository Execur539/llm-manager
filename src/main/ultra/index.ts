/**
 * Ultra: an effort level above the one the model was trained to offer.
 *
 * Every other stop on the effort scale is a value the chat template already accepts. This one is
 * not — it is built out of the runtime instead, from two mechanisms that need no cooperation
 * from the model:
 *
 *   Budget forcing.  A reasoning model stops thinking when it judges it has thought enough. That
 *     judgement is a trained habit, not a limit, and it can be overruled: when the answer arrives
 *     on a thinking floor that has not been met, the model's own reasoning is handed back to it
 *     reopened, with a nudge appended, and it carries on from inside its own thought. Repeated
 *     until the floor is met or the continuation budget runs out. This is the technique from the
 *     s1 test-time-scaling work, and the reason it earns its place is that the second pass is
 *     usually where a model notices what the first one assumed.
 *
 *   Best-of-N.  Several independent attempts, then a pass that reads them together and writes
 *     the answer. Diversity comes from temperature, stepped per sample, so each attempt is a
 *     genuine try at the same question rather than a different question.
 *
 * Samples run one after another, never in parallel. The runtime holds a single slot by design
 * (`--parallel 1` in runtime/llama.ts, and every request serialised through `enqueue`), so
 * overlapping them would not be faster — it would divide the context budget and, on hybrid
 * models, allocate per-slot recurrent state until it OOMs.
 */

import { llama, estimateTokens, type ChatMessage, type CompletionOptions } from '../runtime/llama'
import { logger } from '../log'

export interface UltraConfig {
  samples: number
  thinkingFactor: number
  maxContinuations: number
}

export interface UltraSample {
  index: number
  answer: string
  reasoning: string
  /** How many times this sample's thinking was forced onward. */
  continuations: number
  temperature: number
  ms: number
}

export interface UltraEvents {
  /** A sample is about to start. */
  onSampleStart?: (index: number, total: number) => void
  /** Text arriving for the sample currently running. */
  onSampleDelta?: (index: number, text: string) => void
  onSampleReasoning?: (index: number, text: string) => void
  /** A sample finished and is final. */
  onSample?: (sample: UltraSample) => void
  /** The synthesis pass has begun; deltas after this belong to the real answer. */
  onSynthesisStart?: (samples: UltraSample[]) => void
}

/**
 * Temperatures for a run of N samples.
 *
 * Starts below the model's default and climbs: the first attempt should be the one the model
 * most believes, and later ones exist to disagree with it. All-identical sampling would make
 * best-of-N an expensive way to ask the same question N times.
 */
export function sampleTemperatures(n: number): number[] {
  if (n <= 1) return [0.6]
  const lo = 0.45
  const hi = 0.95
  return Array.from({ length: n }, (_, i) => Number((lo + ((hi - lo) * i) / (n - 1)).toFixed(2)))
}

/**
 * Reasoning below this is treated as complete rather than cut short.
 *
 * Roughly a couple of paragraphs. A model that stopped here did so because the question was
 * easy, not because it ran out of patience — "Hello" earns no more thought however hard it is
 * pushed.
 */
const MIN_REASONING_TO_FORCE = 120

/**
 * Does this text read as a model vamping rather than thinking?
 *
 * Forced past the end of what it has to say, a model falls back on the *shape* of reasoning:
 * many short sentences, all opening the same way, each announcing an intention to check
 * something. The content varies, so comparing whole lines finds nothing; the giveaway is that
 * they all start alike. Requires a decent number of sentences before judging, so a genuinely
 * methodical passage of four or five steps is not mistaken for a loop.
 */
export function looksDegenerate(text: string): boolean {
  const sentences = text
    .split(/[\n.!?]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 8)

  if (sentences.length < 8) return false

  const counts = new Map<string, number>()
  for (const s of sentences) {
    const opening = s.toLowerCase().split(/\s+/).slice(0, 2).join(' ')
    counts.set(opening, (counts.get(opening) ?? 0) + 1)
  }

  const commonest = Math.max(...counts.values())
  return commonest / sentences.length > 0.6
}

/** Nudges used to reopen a thought. Varied so a forced continuation does not loop on a phrase. */
const CONTINUATIONS = [
  'Wait — before answering, let me check that.',
  'Hold on. Let me verify the step I just took.',
  'Actually, let me consider whether there is a case I have missed.',
  'Let me re-examine the assumption behind that.'
]

/**
 * Run one sample, forcing its thinking past where the model would have stopped.
 *
 * The floor is expressed as a multiple of the model's own first attempt rather than an absolute
 * token count: what counts as "thinking properly" differs by an order of magnitude between a
 * one-line question and a hard one, and the model's unprompted length is the only estimate of
 * that available before the work is done.
 */
export async function forcedSample(
  base: CompletionOptions,
  cfg: UltraConfig,
  temperature: number,
  index: number,
  events: UltraEvents = {}
): Promise<UltraSample> {
  const startedAt = Date.now()
  let answer = ''
  let reasoning = ''
  let continuations = 0

  const run = async (messages: ChatMessage[]): Promise<{ text: string; thought: string }> => {
    let text = ''
    let thought = ''
    for await (const ev of llama.streamEvents({ ...base, messages, temperature })) {
      if (ev.type === 'text') {
        text += ev.text
        events.onSampleDelta?.(index, ev.text)
      } else if (ev.type === 'reasoning') {
        thought += ev.text
        events.onSampleReasoning?.(index, ev.text)
      }
    }
    return { text, thought }
  }

  const first = await run(base.messages)
  answer = first.text
  reasoning = first.thought

  // Nothing to extend: a model with no separate reasoning channel gets best-of-N only.
  if (!reasoning.trim()) {
    return { index, answer, reasoning, continuations, temperature, ms: Date.now() - startedAt }
  }

  /*
   * A short thought is a finished thought.
   *
   * The floor is a multiple of the model's own first attempt, which breaks down at the bottom of
   * the range: asked something trivial the model thinks for a line or two, and multiplying that
   * demands it keep going when there is nothing left to consider. What comes out is not deeper
   * reasoning but filler drawn from the shape of long reasoning — "Let me verify the coset. Let
   * me check the kernel." — on a question that had no cosets in it. Below this threshold the
   * model is taken at its word.
   */
  if (estimateTokens(reasoning) < MIN_REASONING_TO_FORCE) {
    return { index, answer, reasoning, continuations, temperature, ms: Date.now() - startedAt }
  }

  const floor = Math.round(estimateTokens(reasoning) * cfg.thinkingFactor)

  while (
    continuations < cfg.maxContinuations &&
    estimateTokens(reasoning) < floor &&
    !base.signal?.aborted
  ) {
    const nudge = CONTINUATIONS[continuations % CONTINUATIONS.length]

    /*
     * The thought is handed back reopened, so the model continues from inside it rather than
     * starting a fresh reply. llama-server prefills an assistant message that ends the request
     * (`--prefill-assistant` is on by default), which is what makes this possible at all.
     */
    const continued = await run([
      ...base.messages,
      { role: 'assistant', content: `<think>\n${reasoning}\n\n${nudge}` }
    ])

    continuations++

    /*
     * Reasoning arriving as reasoning is the signal that the reopening took.
     *
     * Whether a `<think>` prefill continues the thought or merely starts a reply depends on the
     * template, and when it fails the model answers instead — the continuation comes back as
     * content, on a question it has already answered. Taking that as the new answer replaced a
     * good reply with whatever the second attempt drifted into. There is no way to detect this
     * ahead of time per model, so it is detected here and forcing is abandoned for this sample.
     */
    if (!continued.thought.trim()) {
      logger.info('ultra', `sample ${index + 1}: thinking could not be reopened, keeping the first answer`)
      break
    }

    // Forced past the end of what it had to say, a model starts vamping. The tell is structural:
    // sentence after sentence opening the same way. Keep what came before and stop.
    if (looksDegenerate(continued.thought)) {
      logger.info('ultra', `sample ${index + 1}: forced thinking began repeating, stopping at ${continuations}`)
      break
    }

    reasoning += `\n\n${nudge}\n${continued.thought}`
    // Only a continuation that also reasoned is allowed to revise the answer.
    if (continued.text.trim()) answer = continued.text
  }

  return { index, answer, reasoning, continuations, temperature, ms: Date.now() - startedAt }
}

/**
 * Build the synthesis prompt.
 *
 * The candidates arrive as answers only, never their reasoning: a forced sample's thinking runs
 * to many thousands of tokens, and three of those would exhaust the window before the pass that
 * has to read them could start.
 */
export function synthesisMessages(original: ChatMessage[], samples: UltraSample[]): ChatMessage[] {
  const candidates = samples
    .map((s, i) => `--- Candidate ${i + 1} ---\n${s.answer.trim() || '(no answer produced)'}`)
    .join('\n\n')

  const instruction =
    `You produced ${samples.length} independent answers to the question above. They are below.\n\n` +
    `${candidates}\n\n` +
    'Write the final answer. Where they agree, that agreement is probably right. Where they ' +
    'disagree, decide which is correct and say so rather than splitting the difference — a ' +
    'blend of a right answer and a wrong one is wrong. Take the strongest reasoning and the ' +
    'best examples from any of them, correct anything all of them got wrong, and answer the ' +
    'original question directly. Do not mention the candidates, the comparison, or that ' +
    'multiple attempts were made; write the answer as though it were the only one.'

  return [...original, { role: 'user', content: instruction }]
}

/**
 * Run the whole Ultra pipeline for a plain (non-tool) turn.
 *
 * Returns the synthesised answer's message list, ready to stream. The caller streams the final
 * pass itself so the answer arrives token by token like any other.
 */
export async function ultraSamples(
  base: CompletionOptions,
  cfg: UltraConfig,
  events: UltraEvents = {}
): Promise<{ samples: UltraSample[]; synthesis: ChatMessage[] }> {
  const count = Math.max(1, Math.min(8, Math.round(cfg.samples)))
  const temps = sampleTemperatures(count)
  const samples: UltraSample[] = []

  for (let i = 0; i < count; i++) {
    if (base.signal?.aborted) break
    events.onSampleStart?.(i, count)
    const sample = await forcedSample(base, cfg, temps[i], i, events)
    samples.push(sample)
    events.onSample?.(sample)
    logger.info(
      'ultra',
      `sample ${i + 1}/${count} at temp ${sample.temperature}: ${sample.continuations} continuation(s), ${sample.ms}ms`
    )
  }

  events.onSynthesisStart?.(samples)
  return { samples, synthesis: synthesisMessages(base.messages, samples) }
}

/**
 * Choose between competing plans for an agent turn.
 *
 * Kept separate from `synthesisMessages` because the job is different: those candidates are
 * answers to be merged into a better answer, while these are courses of action, and the right
 * output is one of them — improved, but still a single coherent sequence. Merging two plans step
 * by step produces something neither sample would have endorsed.
 */
export function planSynthesisMessages(request: string, plans: string[]): ChatMessage[] {
  const listed = plans
    .map((p, i) => `--- Plan ${i + 1} ---\n${p.trim() || '(no plan produced)'}`)
    .join('\n\n')

  return [
    {
      role: 'user',
      content:
        `A task was investigated ${plans.length} times independently, producing these plans.\n\n` +
        `Task:\n${request}\n\n${listed}\n\n` +
        'Decide the best course of action. Take whichever plan is soundest as the basis rather ' +
        'than interleaving them — a sequence assembled from parts of several is one nobody ' +
        'checked end to end — and fold in any step the others got right that it missed. Drop ' +
        'steps that rest on something a plan assumed rather than verified. Reply with the plan ' +
        'itself as short numbered steps and nothing else: no preamble, no mention of the ' +
        'alternatives, no commentary on the comparison.'
    }
  ]
}

/** The guidance handed to the run that does the work for real. */
export function planPreamble(plan: string): string {
  return (
    'Approach settled on after investigating this task several times over. Follow it, but it ' +
    'was written before any of the work was done — if a tool result contradicts it, trust the ' +
    'tool result and say what changed.\n\n' +
    `${plan.trim()}`
  )
}
