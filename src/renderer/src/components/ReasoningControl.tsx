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

export type Choice = string | 'off' | null

export default function ReasoningControl({
  support,
  value,
  onChange,
  disabled = false
}: {
  support: ReasoningSupport | undefined | null
  /** null means "whatever the model does by default". */
  value: Choice
  onChange: (next: Choice) => void
  disabled?: boolean
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
  const stops: Choice[] = ['off', ...support.levels]
  const effective = value ?? support.defaultLevel ?? support.levels.at(-1) ?? null
  const index = Math.max(0, stops.indexOf(effective))
  const max = stops.length - 1
  // 0..1 in the thumb centre's coordinate space; the CSS converts it. See .slider-fill.
  const pct = max > 0 ? index / max : 0

  const current = stops[index]
  const name = current === 'off' ? 'Off' : levelLabel(String(current))
  const hint =
    current === 'off'
      ? 'No thinking. Answers immediately.'
      : (LEVEL_HINTS[String(current)] ?? 'Effort level defined by this model.')

  return (
    <div className="reasoning" data-testid="reasoning-control" data-kind="effort">
      <span className="reasoning-label">Effort</span>

      <div className="slider" title={hint}>
        {/* The filled portion and the stop marks are painted behind the input. */}
        <div className="slider-track">
          <div className="slider-fill" style={{ '--pct': pct } as React.CSSProperties} />
          {stops.map((stop, i) => (
            <span
              key={String(stop)}
              className={`slider-stop${i <= index ? ' passed' : ''}`}
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

      <span className="reasoning-value" data-testid="reasoning-value">
        {name}
      </span>
    </div>
  )
}
