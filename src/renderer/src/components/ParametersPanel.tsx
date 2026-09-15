/**
 * The per-request parameters, in a panel on the right of Chat and Agent.
 *
 * Everything here travels with each request, so a change applies from the next message with no
 * reload. A field left empty sends nothing and shows, as its placeholder, the default that applies
 * instead: the value the app always sends for temperature and top-p, and otherwise what the loaded
 * model's server reports — which already includes any settings the model file recommends.
 */

import { useEffect, useRef, useState } from 'react'
import type { AppSettings } from '@shared/types'
import {
  AGENT_TEMPERATURE,
  DEFAULT_SAMPLING,
  DEFAULT_TEMPERATURE,
  DEFAULT_TOP_P,
  SAMPLING_FIELDS,
  clampSampling,
  type SamplingField,
  type SamplingGroup
} from '@shared/sampling'
import { invoke } from '../lib/api'
import Icon from './Icon'

type Generation = AppSettings['generation']

const OPEN_KEY = 'llmm.paramsOpen'

/**
 * The last parameters this window saw.
 *
 * Chat and Agent each mount their own panel, and a change is saved a moment after it is made. Held
 * here, the panel opened next shows that change straight away instead of reading settings back
 * before the save has landed.
 */
let cached: Generation | null = null

/** Whether the panel is showing: remembered between launches, and shared by Chat and Agent. */
export function useParamsPanel(): [boolean, (open: boolean) => void] {
  const [open, setOpen] = useState<boolean>(() => {
    try {
      return localStorage.getItem(OPEN_KEY) === '1'
    } catch {
      return false
    }
  })
  const set = (next: boolean): void => {
    setOpen(next)
    try {
      localStorage.setItem(OPEN_KEY, next ? '1' : '0')
    } catch {
      // Storage blocked; the panel still opens, it just will not be remembered.
    }
  }
  return [open, set]
}

/** The header button that shows and hides the panel. */
export function ParamsToggle({ open, onToggle }: { open: boolean; onToggle: () => void }): JSX.Element {
  return (
    <button
      className={`params-toggle${open ? ' active' : ''}`}
      onClick={onToggle}
      aria-pressed={open}
      title={open ? 'Hide the model parameters' : 'Model parameters: temperature, sampling and more'}
      data-testid="params-toggle"
    >
      <Icon name="sliders" size={14} />
      Parameters
    </button>
  )
}

const GROUPS: { id: SamplingGroup; title: string }[] = [
  { id: 'sampling', title: 'Sampling' },
  { id: 'length', title: 'Length and seed' },
  { id: 'repetition', title: 'Repetition' },
  { id: 'advanced', title: 'Advanced sampling' }
]

/** A number as a field shows it, without floating-point noise. */
function fmt(n: number, field: SamplingField): string {
  return field.integer ? String(Math.round(n)) : String(Number(n.toFixed(4)))
}

/** A default as an empty field shows it, with llama.cpp's stand-ins for "none" put into words. */
function describeDefault(value: number | null, field: SamplingField): string {
  if (value === null) return 'default'
  if (field.key === 'maxTokens' && value < 0) return 'no limit'
  if (field.key === 'seed' && (value < 0 || value >= 4_294_967_295)) return 'random'
  return fmt(value, field)
}

export default function ParametersPanel({
  kind,
  modelId,
  onClose
}: {
  kind: 'chat' | 'agent'
  /** The loaded model, whose defaults the empty fields show; null when none is loaded. */
  modelId: string | null
  onClose: () => void
}): JSX.Element {
  const [generation, setGeneration] = useState<Generation | null>(cached)
  const [serverDefaults, setServerDefaults] = useState<Record<string, number> | null>(null)
  const pending = useRef<Generation | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (cached) return
    void invoke<AppSettings>('settings:get')
      .then((s) => {
        cached = s.generation
        setGeneration(s.generation)
      })
      .catch(() => undefined)
  }, [])

  // The model decides the defaults, so they are asked for again whenever a different one loads.
  useEffect(() => {
    if (!modelId) {
      setServerDefaults(null)
      return
    }
    let cancelled = false
    void invoke<Record<string, number> | null>('model:sampling-defaults')
      .then((d) => {
        if (!cancelled) setServerDefaults(d)
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [modelId])

  /** Saved a moment after the last change rather than at every step of a slider. */
  const flush = (): void => {
    if (timer.current) clearTimeout(timer.current)
    timer.current = null
    if (pending.current) void invoke('settings:patch', { generation: pending.current }).catch(() => undefined)
    pending.current = null
  }
  // A change still waiting when the panel closes is saved rather than lost.
  useEffect(() => flush, [])

  /*
   * Each change applies to the latest parameters, not to the copy this render saw.
   *
   * Two changes can land before React renders again — a box committing on blur as a slider starts
   * to move — and building each from the render's copy let the second quietly undo the first.
   */
  const update = (change: (current: Generation) => Generation): void => {
    const base = cached ?? generation
    if (!base) return
    const next = change(base)
    cached = next
    setGeneration(next)
    pending.current = next
    if (timer.current) clearTimeout(timer.current)
    timer.current = setTimeout(flush, 350)
  }

  if (!generation) return <aside className="params-panel" data-testid="params-panel" />

  const sampling = generation.sampling
  const setField = (key: SamplingField['key'], value: number | null): void =>
    update((g) => ({ ...g, sampling: { ...g.sampling, [key]: value } }))

  /** What a field is when left empty. */
  const defaultFor = (field: SamplingField): number | null => {
    if (field.key === 'temperature') return kind === 'agent' ? AGENT_TEMPERATURE : DEFAULT_TEMPERATURE
    if (field.key === 'topP') return DEFAULT_TOP_P
    const value = serverDefaults?.[field.serverKey]
    return typeof value === 'number' ? value : null
  }

  const anySet = SAMPLING_FIELDS.some((f) => sampling[f.key] !== null) || sampling.stop.length > 0

  return (
    <aside className="params-panel" data-testid="params-panel">
      <div className="params-head">
        <span className="params-title">Parameters</span>
        <button
          className="subtle"
          disabled={!anySet}
          onClick={() => update((g) => ({ ...g, sampling: { ...DEFAULT_SAMPLING, stop: [] } }))}
          title="Clear every parameter back to its default"
          data-testid="params-reset"
        >
          Reset
        </button>
        <button className="icon" onClick={onClose} aria-label="Hide parameters" title="Hide parameters">
          <Icon name="close" size={13} />
        </button>
      </div>
      <p className="params-note">
        Applies from the next message, with no reload. Leave a field empty to use the default shown in it.
        {!modelId && ' Load a model to see its own defaults.'}
      </p>

      {kind === 'chat' ? (
        <section className="params-group">
          <div className="params-group-title">System prompt</div>
          <textarea
            className="params-text"
            rows={4}
            value={generation.systemPrompt}
            placeholder="Instructions every chat starts from, such as a role or a tone"
            onChange={(e) => {
              const systemPrompt = e.target.value
              update((g) => ({ ...g, systemPrompt }))
            }}
            data-testid="params-system-prompt"
          />
        </section>
      ) : (
        <p className="params-note">
          The agent builds its own system prompt around its tools, so only the sampling parameters apply here.
        </p>
      )}

      {GROUPS.map((group) => {
        const rows = (
          <>
            {SAMPLING_FIELDS.filter((f) => f.group === group.id).map((f) => (
              <ParamRow
                key={f.key}
                field={f}
                value={sampling[f.key]}
                fallback={defaultFor(f)}
                onChange={(v) => setField(f.key, v)}
              />
            ))}
            {group.id === 'length' && (
              <StopStrings value={sampling.stop} onChange={(stop) => update((g) => ({ ...g, sampling: { ...g.sampling, stop } }))} />
            )}
          </>
        )
        return group.id === 'advanced' ? (
          <details key={group.id} className="params-group">
            <summary className="params-group-title">{group.title}</summary>
            {rows}
          </details>
        ) : (
          <section key={group.id} className="params-group">
            <div className="params-group-title">{group.title}</div>
            {rows}
          </section>
        )
      })}
    </aside>
  )
}

/** One parameter: a box to type in, a slider where the range suits one, and a way back to the default. */
function ParamRow({
  field,
  value,
  fallback,
  onChange
}: {
  field: SamplingField
  value: number | null
  fallback: number | null
  onChange: (value: number | null) => void
}): JSX.Element {
  const [text, setText] = useState(value === null ? '' : fmt(value, field))
  // Follows the stored value when it changes underneath: the slider, a reset, another panel.
  useEffect(() => setText(value === null ? '' : fmt(value, field)), [value, field])

  const commit = (): void => {
    const typed = text.trim()
    if (!typed) {
      if (value !== null) onChange(null)
      return
    }
    const next = clampSampling(field, Number(typed))
    if (next === null) {
      setText(value === null ? '' : fmt(value, field))
      return
    }
    setText(fmt(next, field))
    if (next !== value) onChange(next)
  }

  const id = `param-${field.key}`
  return (
    <div className={`param${value === null ? '' : ' set'}`} title={field.hint} data-testid={id}>
      <div className="param-head">
        <label className="param-label" htmlFor={id}>
          {field.label}
        </label>
        {field.choices ? (
          <select
            id={id}
            className="param-select"
            value={value === null ? '' : String(value)}
            onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
          >
            <option value="">
              {fallback === null
                ? 'Default'
                : `Default (${field.choices.find((c) => c.value === fallback)?.label ?? fallback})`}
            </option>
            {field.choices.map((c) => (
              <option key={c.value} value={c.value}>
                {c.label}
              </option>
            ))}
          </select>
        ) : (
          <input
            id={id}
            className="param-input"
            type="text"
            inputMode="decimal"
            value={text}
            placeholder={describeDefault(fallback, field)}
            onChange={(e) => setText(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => {
              if (e.key === 'Enter') e.currentTarget.blur()
            }}
          />
        )}
        <button
          className="param-reset"
          onClick={() => onChange(null)}
          disabled={value === null}
          aria-label={`Reset ${field.label}`}
          title="Back to the default"
        >
          <Icon name="close" size={11} />
        </button>
      </div>
      {field.slider && (
        <input
          type="range"
          className={`param-slider${value === null ? ' unset' : ''}`}
          min={field.min}
          max={field.max}
          step={field.step}
          value={value ?? fallback ?? field.min}
          onChange={(e) => onChange(clampSampling(field, Number(e.target.value)))}
          aria-label={field.label}
        />
      )}
    </div>
  )
}

/** Stop strings, one per line. `\n` and `\t` stand for a newline and a tab. */
function StopStrings({ value, onChange }: { value: string[]; onChange: (stop: string[]) => void }): JSX.Element {
  const show = (list: string[]): string => list.map((s) => s.replace(/\n/g, '\\n').replace(/\t/g, '\\t')).join('\n')
  const [text, setText] = useState(show(value))
  useEffect(() => setText(show(value)), [value])

  const commit = (): void => {
    const next = text
      .split('\n')
      .map((line) => line.replace(/\\n/g, '\n').replace(/\\t/g, '\t'))
      .filter((s) => s.length > 0)
    if (JSON.stringify(next) !== JSON.stringify(value)) onChange(next)
  }

  return (
    <div
      className={`param${value.length ? ' set' : ''}`}
      title="Generation stops as soon as any of these appears."
      data-testid="param-stop"
    >
      <div className="param-head">
        <label className="param-label" htmlFor="param-stop-text">
          Stop strings
        </label>
        <button
          className="param-reset"
          onClick={() => onChange([])}
          disabled={!value.length}
          aria-label="Clear stop strings"
          title="Clear"
        >
          <Icon name="close" size={11} />
        </button>
      </div>
      <textarea
        id="param-stop-text"
        className="params-text"
        rows={2}
        value={text}
        placeholder={'One per line, \\n for a newline'}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
      />
    </div>
  )
}
