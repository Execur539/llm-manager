/**
 * Reasoning effort control.
 *
 * Shown only when the loaded model's chat template actually offers one, and shaped to match what
 * it offers: a stepped slider when the template enumerates effort levels, a switch when it only
 * honours `enable_thinking`, and nothing at all otherwise. The level names come from the model —
 * Qwen3.8 calls its strongest setting `xhigh` while gpt-oss calls it `high` — so nothing here
 * assumes a fixed vocabulary.
 *
 * Built on a native range input. Styling one is more work than drawing a div, but it is the only
 * way to get keyboard stepping, page-up/down, home/end and screen-reader semantics for free on a
 * control whose whole purpose is picking one of a few ordered values.
 */

import { useEffect, useRef, useState } from 'react'
import type { ReasoningSupport } from '@shared/types'

/** Human wording for the level names models actually use. */
const LEVEL_LABELS: Record<string, string> = {
  minimal: 'Minimal',
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  xhigh: 'Extra high',
  max: 'Max'
}

/** One line on what each level costs, where the name alone does not say. */
const LEVEL_HINTS: Record<string, string> = {
  minimal: 'Barely thinks. Fastest, weakest on anything multi-step.',
  low: 'Brief reasoning, straight to the answer.',
  medium: 'Balanced — thinks through the problem without labouring it.',
  high: 'Works carefully through the problem. Slower.',
  xhigh: 'Validates assumptions and considers alternatives. Slowest.',
  max: 'Everything it has. Very slow.'
}

export function levelLabel(level: string): string {
  return LEVEL_LABELS[level] ?? level.replace(/^\w/, (c) => c.toUpperCase())
}

/** Kept in step with main/models/reasoning.ts, which defines the same constant. */
export const ULTRA = 'ultra'

export type Choice = string | 'off' | typeof ULTRA | null

export default function ReasoningControl({
  support,
  value,
  onChange,
  disabled = false,
  samples = 3,
  onSamplesChange
}: {
  support: ReasoningSupport | undefined | null
  /** null means "whatever the model does by default". */
  value: Choice
  onChange: (next: Choice) => void
  disabled?: boolean
  /** Ultra's attempt count, surfaced here because this is where Ultra is chosen. */
  samples?: number
  onSamplesChange?: (next: number) => void
}): JSX.Element | null {
  if (!support || support.kind === 'none') return null

  // ---- a model with only an on/off switch
  if (support.kind === 'toggle') {
    const on = value !== 'off'
    return (
      <div className="reasoning" data-testid="reasoning-control" data-kind="toggle">
        <span className="reasoning-label">Thinking</span>
        <button
          type="button"
          role="switch"
          aria-checked={on}
          aria-label="Thinking"
          className={`switch${on ? ' on' : ''}`}
          disabled={disabled}
          onClick={() => onChange(on ? 'off' : 'on')}
          title={on ? 'Thinking on — click to turn off' : 'Thinking off — click to turn on'}
          data-testid="reasoning-toggle"
        >
          <span className="switch-knob" />
        </button>
      </div>
    )
  }

  // ---- a model with named effort levels
  //
  // "Off" occupies position 0, so the whole range is one continuous control rather than a slider
  // plus a separate switch. It is offered for every reasoning model, not only those whose
  // template has a switch of its own: llama.cpp can end the thought block itself, which covers
  // the effort-only templates that enumerate levels without providing a way to say "none".
  //
  // "Ultra" occupies the last position, past everything the template offers. It is not a level
  // the model knows — no template has one — but a runtime mode built on top of its strongest
  // native setting, so it belongs at the end of the same scale rather than in a control of its
  // own. It is marked so it never passes for something the model advertised.
  const stops: Choice[] = ['off', ...support.levels, ULTRA]
  const effective = value ?? support.defaultLevel ?? support.levels.at(-1) ?? null
  const index = Math.max(0, stops.indexOf(effective))
  const max = stops.length - 1

  const current = stops[index]
  const isUltra = current === ULTRA
  const name = current === 'off' ? 'Off' : isUltra ? 'Ultra' : levelLabel(String(current))
  const hint = isUltra
    ? `Beyond the model's own maximum: ${samples} independent attempts, each pushed to keep thinking past where it would have stopped, then compared into one answer. Several times slower.`
    : current === 'off'
      ? 'No thinking. Answers immediately.'
      : (LEVEL_HINTS[String(current)] ?? 'Effort level defined by this model.')

  // The level the model falls back to on its own, marked so the scale has a reference point.
  const defaultIndex = support.defaultLevel ? stops.indexOf(support.defaultLevel) : -1

  return (
    <EffortPopover
      name={name}
      hint={hint}
      disabled={disabled}
      panel={
        <>
          <div className="reasoning-head">
            <span className="reasoning-label">Effort</span>
            <span className="reasoning-value" data-testid="reasoning-value">
              {name}
            </span>
          </div>

          {/* Naming both ends says what the axis costs, which the level names alone do not. */}
          <div className="reasoning-anchors" aria-hidden="true">
            <span>Faster</span>
            <span>Smarter</span>
          </div>

          <div className="slider">
            {/* One mark per value. No filled bar: the stops are the scale. */}
            <div className="slider-track">
              {stops.map((stop, i) => (
                <span
                  key={String(stop)}
                  className={
                    'slider-stop' +
                    (i === index ? ' current' : '') +
                    (i === defaultIndex ? ' is-default' : '') +
                    (stop === ULTRA ? ' is-ultra' : '')
                  }
                  style={{ '--stop': max > 0 ? i / max : 0 } as React.CSSProperties}
                />
              ))}
            </div>
            <input
              type="range"
              min={0}
              max={max}
              step={1}
              value={index}
              disabled={disabled}
              onChange={(e) => onChange(stops[Number(e.target.value)] ?? null)}
              aria-label="Reasoning effort"
              aria-valuetext={name}
              data-testid="reasoning-slider"
            />
          </div>

          {/* Only Ultra spends this, so the cost only appears when it is about to be spent. */}
          {isUltra && (
            <label className="ultra-config" data-testid="ultra-samples-config">
              <span>Attempts</span>
              <input
                type="number"
                min={1}
                max={8}
                value={samples}
                disabled={disabled}
                onChange={(e) => onSamplesChange?.(clampSamples(Number(e.target.value)))}
                data-testid="ultra-samples-input"
              />
            </label>
          )}
        </>
      }
    />
  )
}

/** Bounded where the engine bounds it, so the field cannot ask for something it will not get. */
export function clampSamples(n: number): number {
  return Number.isFinite(n) ? Math.max(1, Math.min(8, Math.round(n))) : 3
}

/**
 * The effort scale, folded away behind the level it has chosen.
 *
 * The composer's meta row is a strip of small print, and a three-row control with its own axis
 * labels does not belong there permanently — the setting is glanced at far more often than it is
 * changed. So the row carries only the current level, and the scale opens over it on demand.
 *
 * Deliberately not a <details>: it has to escape the composer's bounds, and it needs to close on
 * Escape and on a click elsewhere, neither of which that element does.
 */
function EffortPopover({
  name,
  hint,
  disabled,
  panel
}: {
  name: string
  hint: string
  disabled: boolean
  panel: React.ReactNode
}): JSX.Element {
  const [open, setOpen] = useState(false)
  const root = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return

    const onDown = (e: PointerEvent): void => {
      if (!root.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') {
        e.stopPropagation()
        setOpen(false)
      }
    }
    // Captured, so dragging the slider inside the panel never counts as a click outside it.
    document.addEventListener('pointerdown', onDown, true)
    document.addEventListener('keydown', onKey, true)
    return () => {
      document.removeEventListener('pointerdown', onDown, true)
      document.removeEventListener('keydown', onKey, true)
    }
  }, [open])

  // A setting that cannot be changed should not offer to open.
  useEffect(() => {
    if (disabled) setOpen(false)
  }, [disabled])

  return (
    <div className="reasoning" data-testid="reasoning-control" data-kind="effort" ref={root}>
      <button
        type="button"
        className={`reasoning-trigger${open ? ' open' : ''}`}
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        aria-expanded={open}
        aria-haspopup="dialog"
        title={`Reasoning effort — ${hint}`}
        data-testid="reasoning-trigger"
      >
        {name}
      </button>

      {open && (
        <div className="reasoning-pop" role="dialog" aria-label="Reasoning effort" data-testid="reasoning-pop">
          {panel}
          <p className="reasoning-hint">{hint}</p>
        </div>
      )}
    </div>
  )
}
