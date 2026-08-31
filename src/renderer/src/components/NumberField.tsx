import { useEffect, useState } from 'react'

/**
 * A number input that only ever hands its caller a number.
 *
 * Settings fields were bound straight to a save on every keystroke, through
 * `Number(e.target.value)`. That writes on each intermediate state, and two of the states a
 * `type="number"` box passes through are not numbers: an empty field is `0`, and a lone minus
 * sign is `NaN` — which `JSON.stringify` stores as `null`.
 *
 * For most fields that is untidy. For the agent's tool-call ceiling it was fatal: the loop is
 * `while (calls < max)`, and both `0 < 0` and `0 < null` are false, so clearing the box to
 * retype it left every agent turn ending instantly, with no answer and no error, across
 * restarts, until somebody thought to look at Settings.
 *
 * Typing stays free-form in local state. The value is clamped and committed on blur — or on
 * Enter, which blurs — so the stored setting is never an intermediate keystroke.
 */
export default function NumberField({
  value,
  min,
  max,
  onCommit,
  hint,
  disabled,
  testId
}: {
  value: number
  min: number
  max: number
  onCommit: (n: number) => void
  hint?: string
  disabled?: boolean
  testId?: string
}): JSX.Element {
  const [text, setText] = useState(String(value))

  // Follow the stored value when it changes underneath us — a reset, or another window.
  useEffect(() => setText(String(value)), [value])

  const commit = (): void => {
    const parsed = Number(text)
    const next = Number.isFinite(parsed) ? Math.min(max, Math.max(min, Math.round(parsed))) : value
    setText(String(next))
    if (next !== value) onCommit(next)
  }

  return (
    <>
      <input
        type="number"
        min={min}
        max={max}
        value={text}
        disabled={disabled}
        data-testid={testId}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') e.currentTarget.blur()
        }}
      />
      {hint && (
        <span className="faint" style={{ marginLeft: 6 }}>
          {hint}
        </span>
      )}
    </>
  )
}
