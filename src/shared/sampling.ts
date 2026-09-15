/**
 * Per-request generation parameters: the ones that travel with each request to llama-server, so
 * they can change between messages without reloading the model.
 *
 * Shared by the main process, which puts them on requests, and the parameters panel, which draws a
 * control for each — so the two agree on every name, range and default. A null setting means "not
 * set": the request leaves the field out and the model's own default applies, except for the two
 * the app has always sent a value for (DEFAULT_TEMPERATURE and DEFAULT_TOP_P).
 */

export type SamplingKey =
  | 'temperature'
  | 'topP'
  | 'topK'
  | 'minP'
  | 'typicalP'
  | 'topNSigma'
  | 'repeatPenalty'
  | 'repeatLastN'
  | 'presencePenalty'
  | 'frequencyPenalty'
  | 'dryMultiplier'
  | 'dryBase'
  | 'dryAllowedLength'
  | 'dryPenaltyLastN'
  | 'xtcProbability'
  | 'xtcThreshold'
  | 'mirostat'
  | 'mirostatTau'
  | 'mirostatEta'
  | 'maxTokens'
  | 'seed'

/** The parameters as stored: every one of them, null where nobody has set it. */
export type SamplingSettings = { [K in SamplingKey]: number | null } & { stop: string[] }

/** The same parameters as a request carries them: only those that are set. */
export type SamplingOptions = { [K in SamplingKey]?: number } & { stop?: string[] }

/** Sent when nothing else is — what the app has always used for chat, the API and everything else. */
export const DEFAULT_TEMPERATURE = 0.7
export const DEFAULT_TOP_P = 0.95
/** The agent runs cooler by default: a tool call rewards the likeliest continuation. */
export const AGENT_TEMPERATURE = 0.6

export type SamplingGroup = 'sampling' | 'length' | 'repetition' | 'advanced'

export interface SamplingField {
  key: SamplingKey
  label: string
  group: SamplingGroup
  /** The field's name in a llama-server request, and in the defaults its `/props` reports. */
  serverKey: string
  min: number
  max: number
  step: number
  integer?: boolean
  /** Whether a slider suits the range. Wide integer ranges get a box alone. */
  slider?: boolean
  /** Fixed choices, for a parameter that is a mode rather than a quantity. */
  choices?: { value: number; label: string }[]
  hint: string
}

export const SAMPLING_FIELDS: SamplingField[] = [
  {
    key: 'temperature', label: 'Temperature', group: 'sampling', serverKey: 'temperature',
    min: 0, max: 2, step: 0.05, slider: true,
    hint: 'Randomness. Lower is more focused and repeatable, higher more varied.'
  },
  {
    key: 'topP', label: 'Top P', group: 'sampling', serverKey: 'top_p',
    min: 0, max: 1, step: 0.01, slider: true,
    hint: 'Samples only from the likeliest tokens whose probabilities add up to this. 1 turns it off.'
  },
  {
    key: 'topK', label: 'Top K', group: 'sampling', serverKey: 'top_k',
    min: 0, max: 200, step: 1, integer: true, slider: true,
    hint: 'Samples only from this many of the likeliest tokens. 0 turns it off.'
  },
  {
    key: 'minP', label: 'Min P', group: 'sampling', serverKey: 'min_p',
    min: 0, max: 1, step: 0.01, slider: true,
    hint: 'Drops tokens less likely than this fraction of the likeliest one. 0 turns it off.'
  },
  {
    key: 'maxTokens', label: 'Max tokens', group: 'length', serverKey: 'max_tokens',
    min: 1, max: 1_000_000, step: 1, integer: true,
    hint: 'The longest a reply may be, in tokens, thinking included. Empty is no limit.'
  },
  {
    key: 'seed', label: 'Seed', group: 'length', serverKey: 'seed',
    min: -1, max: 4_294_967_295, step: 1, integer: true,
    hint: 'The same seed and settings give the same reply to the same prompt. -1 is random.'
  },
  {
    key: 'repeatPenalty', label: 'Repeat penalty', group: 'repetition', serverKey: 'repeat_penalty',
    min: 0.5, max: 2, step: 0.01, slider: true,
    hint: 'Discourages repeating recent tokens. 1 turns it off.'
  },
  {
    key: 'repeatLastN', label: 'Repeat window', group: 'repetition', serverKey: 'repeat_last_n',
    min: -1, max: 131_072, step: 1, integer: true,
    hint: 'How many recent tokens the penalties look back over. 0 turns them off, -1 is the whole context.'
  },
  {
    key: 'presencePenalty', label: 'Presence penalty', group: 'repetition', serverKey: 'presence_penalty',
    min: -2, max: 2, step: 0.05, slider: true,
    hint: 'Penalises any token already used, once. Positive values push towards new ground.'
  },
  {
    key: 'frequencyPenalty', label: 'Frequency penalty', group: 'repetition', serverKey: 'frequency_penalty',
    min: -2, max: 2, step: 0.05, slider: true,
    hint: 'Penalises a token more the more often it has already appeared.'
  },
  {
    key: 'typicalP', label: 'Typical P', group: 'advanced', serverKey: 'typical_p',
    min: 0, max: 1, step: 0.01, slider: true,
    hint: 'Locally typical sampling. 1 turns it off.'
  },
  {
    key: 'topNSigma', label: 'Top-nσ', group: 'advanced', serverKey: 'top_n_sigma',
    min: -1, max: 10, step: 0.1, slider: true,
    hint: 'Keeps tokens within this many standard deviations of the top one. -1 turns it off.'
  },
  {
    key: 'xtcProbability', label: 'XTC probability', group: 'advanced', serverKey: 'xtc_probability',
    min: 0, max: 1, step: 0.01, slider: true,
    hint: 'Chance per token of excluding the top choices, for less predictable writing. 0 turns it off.'
  },
  {
    key: 'xtcThreshold', label: 'XTC threshold', group: 'advanced', serverKey: 'xtc_threshold',
    min: 0, max: 0.5, step: 0.01, slider: true,
    hint: 'How likely a token has to be for XTC to exclude it.'
  },
  {
    key: 'dryMultiplier', label: 'DRY multiplier', group: 'advanced', serverKey: 'dry_multiplier',
    min: 0, max: 5, step: 0.05, slider: true,
    hint: 'Strength of the DRY penalty on repeated passages. 0 turns it off.'
  },
  {
    key: 'dryBase', label: 'DRY base', group: 'advanced', serverKey: 'dry_base',
    min: 1, max: 4, step: 0.05, slider: true,
    hint: 'How fast the DRY penalty grows with the length of a repeat.'
  },
  {
    key: 'dryAllowedLength', label: 'DRY allowed length', group: 'advanced', serverKey: 'dry_allowed_length',
    min: 1, max: 20, step: 1, integer: true, slider: true,
    hint: 'Repeats up to this many tokens long are not penalised.'
  },
  {
    key: 'dryPenaltyLastN', label: 'DRY window', group: 'advanced', serverKey: 'dry_penalty_last_n',
    min: -1, max: 131_072, step: 1, integer: true,
    hint: 'How many recent tokens DRY looks back over. -1 is the whole context.'
  },
  {
    key: 'mirostat', label: 'Mirostat', group: 'advanced', serverKey: 'mirostat',
    min: 0, max: 2, step: 1, integer: true,
    choices: [
      { value: 0, label: 'Off' },
      { value: 1, label: 'Mirostat' },
      { value: 2, label: 'Mirostat 2.0' }
    ],
    hint: 'Aims for a steady level of surprise, in place of Top K, Top P and Min P.'
  },
  {
    key: 'mirostatTau', label: 'Mirostat target', group: 'advanced', serverKey: 'mirostat_tau',
    min: 0, max: 10, step: 0.1, slider: true,
    hint: 'The level of surprise Mirostat aims for. Lower is more focused.'
  },
  {
    key: 'mirostatEta', label: 'Mirostat rate', group: 'advanced', serverKey: 'mirostat_eta',
    min: 0, max: 1, step: 0.01, slider: true,
    hint: 'How quickly Mirostat adjusts towards its target.'
  }
]

/** Nothing set: every request goes out with the defaults. */
export const DEFAULT_SAMPLING: SamplingSettings = {
  ...(Object.fromEntries(SAMPLING_FIELDS.map((f) => [f.key, null])) as { [K in SamplingKey]: null }),
  stop: []
}

/** A value brought inside its field's range, or null when there is no usable number. */
export function clampSampling(field: SamplingField, value: unknown): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null
  const v = Math.min(field.max, Math.max(field.min, value))
  return field.integer ? Math.round(v) : v
}

/**
 * The parameters that are set, as request options.
 *
 * Brought into range, or dropped, rather than trusted: the settings file is hand-editable, and
 * llama-server rejects some values outright.
 */
export function samplingOptions(settings: Partial<SamplingSettings> | null | undefined): SamplingOptions {
  const out: SamplingOptions = {}
  if (!settings) return out
  for (const field of SAMPLING_FIELDS) {
    const value = clampSampling(field, settings[field.key])
    if (value !== null) out[field.key] = value
  }
  const stop = Array.isArray(settings.stop)
    ? settings.stop.filter((s): s is string => typeof s === 'string' && s.length > 0)
    : []
  if (stop.length) out.stop = stop
  return out
}
