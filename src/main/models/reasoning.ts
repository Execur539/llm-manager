/**
 * Working out what reasoning control a model actually offers, from its own chat template.
 *
 * There is no metadata key for this and no standard set of level names — the template is the
 * only honest source. Three shapes exist in the wild:
 *
 *   effort   the template branches on `reasoning_effort` and enumerates the levels it accepts.
 *            Qwen3.8-27B takes low / medium / xhigh (default xhigh); gpt-oss takes low / medium
 *            / high. Hardcoding either list would be wrong for the other, so the levels are read
 *            out of the template.
 *   toggle   the template honours `enable_thinking` but has no levels — Qwen3.6, nanbeige4.2.
 *   none     no reasoning control at all — Hermes-3's template is 209 characters long.
 *
 * llama-server passes `reasoning_effort` and `chat_template_kwargs` straight through to the
 * template, so whatever is found here is exactly what can be sent back.
 */

/** Ordered weakest to strongest. Names outside this list sort after it, alphabetically. */
const CANONICAL_ORDER = ['minimal', 'low', 'medium', 'high', 'xhigh', 'max']

/** Values that mean "no thinking" rather than naming a level of it. */
const OFF_VALUES = new Set(['none', 'off', 'false', 'disabled'])

/** Values that mean "whatever the template already does". */
const PASSTHROUGH_VALUES = new Set(['default', 'auto'])

export type ReasoningKind = 'none' | 'toggle' | 'effort'

export interface ReasoningSupport {
  kind: ReasoningKind
  /** Accepted effort levels, ordered weakest to strongest. Empty unless kind is 'effort'. */
  levels: string[]
  /** What the template falls back to when nothing is passed. */
  defaultLevel: string | null
  /** Whether thinking can be switched off entirely, via enable_thinking or an explicit 'none'. */
  canDisable: boolean
  /**
   * The literal this template accepts as an effort level meaning "do not think", if any.
   *
   * Kept separate from `canDisable` because the two are not the same question. A template can be
   * disabled through `enable_thinking` while still validating `reasoning_effort` against a list
   * that has no off value in it — sending 'none' to that template does not disable anything, it
   * raises. Only a name found in the template's own accepted set is ever sent back to it.
   */
  offValue: string | null
}

export const NO_REASONING: ReasoningSupport = {
  kind: 'none',
  levels: [],
  defaultLevel: null,
  canDisable: false,
  offValue: null
}

/**
 * Names the template has assigned `reasoning_effort` to.
 *
 * Templates rarely branch on the raw variable: they normalise it first, and the name they
 * normalise it into is arbitrary. Qwen3.8 happens to pick `resolved_reasoning_effort`, which a
 * substring match catches by luck; a template using `{%- set r = reasoning_effort|default(...) %}`
 * would be missed entirely, and the model would silently lose its effort control.
 */
function aliasNames(template: string): string[] {
  const names = new Set<string>()
  for (const m of template.matchAll(/\bset\s+([A-Za-z_][A-Za-z0-9_]*)\s*=[^%]*?reasoning_effort/g)) {
    names.add(m[1])
  }
  return [...names]
}

/**
 * A pattern matching the raw variable or any name the template aliased it to.
 *
 * Alias names come from a capture that only accepts identifier characters, so none of them can
 * carry regex metacharacters and no escaping is needed here.
 */
function effortVariablePattern(template: string): string {
  return `(?:${['[A-Za-z_]*reasoning_effort', ...aliasNames(template)].join('|')})`
}

/** Pull the quoted strings out of a Jinja tuple/list literal. */
function quotedStrings(source: string): string[] {
  return [...source.matchAll(/['"]([A-Za-z0-9_-]+)['"]/g)].map((m) => m[1])
}

function rank(level: string): number {
  const i = CANONICAL_ORDER.indexOf(level)
  return i === -1 ? CANONICAL_ORDER.length : i
}

function order(levels: Iterable<string>): string[] {
  return [...new Set(levels)].sort((a, b) => {
    const d = rank(a) - rank(b)
    return d !== 0 ? d : a.localeCompare(b)
  })
}

/** What the user chose. `null` means "leave the template's own default alone". */
export type ReasoningChoice = string | 'off' | typeof ULTRA | null

/**
 * The level above the model's own maximum.
 *
 * Not a level the template knows — no template has one — so it never reaches the model as an
 * effort value. It is a runtime mode, implemented in main/ultra: the request itself goes out at
 * the model's strongest native level, and the extra effort comes from what surrounds it.
 */
export const ULTRA = 'ultra' as const

/** Is this choice the synthetic level rather than one the model named? */
export function isUltra(choice: ReasoningChoice): boolean {
  return choice === ULTRA
}

/**
 * The strongest level the template actually accepts.
 *
 * Ultra runs on top of this: forcing a model to think longer is worth more when it is already
 * thinking as hard as it knows how.
 */
export function strongestLevel(support: ReasoningSupport | undefined | null): string | null {
  if (!support || support.kind !== 'effort') return null
  return support.levels.at(-1) ?? null
}

/**
 * Narrow a remembered choice to something the loaded model can actually express.
 *
 * The effort setting is remembered per conversation while level names belong to the model, so
 * switching models leaves a choice the new one has never heard of. `reasoningRequestFields`
 * already drops an unknown level, but `isUltra` did not ask the same question — so Ultra chosen
 * on a levels model survived onto a toggle model, where the control has no way to show it, and
 * ran three sampling passes with nothing on screen to explain the wait.
 *
 * Applied here rather than only in the renderer because the renderer is not the only caller: the
 * API server and a remote browser send this field too, and neither is obliged to be sensible.
 *
 * The renderer keeps a copy of this rule, because it decides what to *draw* from the same facts;
 * they are checked against each other by construction being three lines long.
 */
export function sendableChoice(support: ReasoningSupport | undefined | null, choice: ReasoningChoice): ReasoningChoice {
  if (!support || support.kind === 'none' || choice === null) return null
  if (choice === 'off') return 'off'
  if (support.kind === 'toggle') return choice === ULTRA ? null : choice
  if (choice === ULTRA) return ULTRA
  return support.levels.includes(choice) ? choice : null
}

/**
 * Turn a choice into request fields, given what the model actually supports.
 *
 * Deliberately conservative: anything the model did not advertise is dropped rather than sent
 * hopefully. A level name a template does not know makes it raise an exception mid-request, and
 * `enable_thinking` on a model that ignores it is silent but misleading in the logs.
 */
export function reasoningRequestFields(
  support: ReasoningSupport | undefined | null,
  choice: ReasoningChoice
): {
  reasoningEffort?: string
  chatTemplateKwargs?: Record<string, unknown>
  reasoningBudget?: number
} {
  if (!support || support.kind === 'none' || choice === null) return {}

  /*
   * Turning thinking off works on any reasoning model, whatever its template offers.
   *
   * Three mechanisms, sent together because which one bites depends on the model:
   *   - `reasoning_budget: 0` is llama.cpp's own, and closes the thought block itself. It is the
   *     only one that works on an effort-only template like Qwen3.8's, which enumerates levels
   *     and has no way to express "none".
   *   - `enable_thinking: false` is honoured by templates built around that flag, and is an
   *     unread variable everywhere else.
   *   - the effort level is sent only when the template's own accepted set contains an off
   *     value. Sending 'none' to a template that validates against ('xhigh','medium','low')
   *     does not disable thinking — it raises mid-request and the turn fails.
   */
  if (choice === 'off') {
    return {
      reasoningBudget: 0,
      chatTemplateKwargs: { enable_thinking: false },
      ...(support.offValue ? { reasoningEffort: support.offValue } : {})
    }
  }

  /*
   * Ultra is not a value any template accepts, so it is never sent as one. The request goes out
   * at the model's strongest native level and the extra effort is applied around it — see
   * main/ultra. Sending the literal 'ultra' would raise on any template that validates its
   * levels, which is the same trap that 'none' used to fall into.
   */
  if (choice === ULTRA) {
    if (support.kind === 'toggle') return { chatTemplateKwargs: { enable_thinking: true } }
    const strongest = strongestLevel(support)
    return strongest ? { reasoningEffort: strongest } : {}
  }

  if (support.kind === 'toggle') {
    // The only meaningful non-off choice for a toggle is "on".
    return { chatTemplateKwargs: { enable_thinking: true } }
  }

  if (!support.levels.includes(choice)) return {}
  return { reasoningEffort: choice }
}

export function detectReasoning(chatTemplate: string | null | undefined): ReasoningSupport {
  if (typeof chatTemplate !== 'string' || chatTemplate.length === 0) return NO_REASONING

  const mentionsEffort = /reasoning_effort/.test(chatTemplate)
  const mentionsToggle = /enable_thinking/.test(chatTemplate)

  if (!mentionsEffort && !mentionsToggle) return NO_REASONING

  let levels: string[] = []
  let canDisable = mentionsToggle
  let offValue: string | null = null

  if (mentionsEffort) {
    /*
     * A membership test is authoritative when present: it is the template's own validation of
     * what it will accept, so it excludes aliases. Qwen3.8 remaps 'high' onto 'xhigh' *before*
     * testing `not in ('xhigh', 'medium', 'low')`, so collecting equality comparisons instead
     * would offer 'high' as a fourth stop that behaves identically to 'xhigh'.
     */
    const variable = effortVariablePattern(chatTemplate)
    const membership = [
      ...chatTemplate.matchAll(new RegExp(`\\b${variable}\\s+(?:not\\s+)?in\\s*(\\([^)]*\\)|\\[[^\\]]*\\])`, 'g'))
    ]
    for (const m of membership) levels.push(...quotedStrings(m[1]))

    if (levels.length === 0) {
      // No validation list — fall back to whatever the template compares against. The variable
      // is often a derived one (`resolved_reasoning_effort`), hence the loose prefix match.
      for (const m of chatTemplate.matchAll(new RegExp(`\\b${variable}\\s*(?:==|!=)\\s*['"]([A-Za-z0-9_-]+)['"]`, 'g'))) {
        levels.push(m[1])
      }
      for (const m of chatTemplate.matchAll(new RegExp(`['"]([A-Za-z0-9_-]+)['"]\\s*(?:==|!=)\\s*\\b${variable}`, 'g'))) {
        levels.push(m[1])
      }
    }

    // 'none' is llama.cpp's way of disabling thinking, not a level of it.
    const off = levels.find((l) => OFF_VALUES.has(l.toLowerCase()))
    if (off) {
      canDisable = true
      // Kept verbatim: the template will compare against its own spelling, not a normalised one.
      offValue = off
    }
    levels = levels.filter((l) => !OFF_VALUES.has(l.toLowerCase()) && !PASSTHROUGH_VALUES.has(l.toLowerCase()))
  }

  const ordered = order(levels)

  // The template's own fallback, e.g. `reasoning_effort|default('xhigh')`.
  const defaultMatch = chatTemplate.match(/reasoning_effort\s*\|\s*default\s*\(\s*['"]([A-Za-z0-9_-]+)['"]/)
  const parsedDefault = defaultMatch?.[1] ?? null
  const defaultLevel = parsedDefault && ordered.includes(parsedDefault) ? parsedDefault : (ordered.at(-1) ?? null)

  // One level is not a choice; treat it as a plain toggle if the template offers one.
  if (ordered.length >= 2) {
    return { kind: 'effort', levels: ordered, defaultLevel, canDisable, offValue }
  }
  if (canDisable) {
    return { kind: 'toggle', levels: [], defaultLevel: null, canDisable: true, offValue }
  }
  return NO_REASONING
}
