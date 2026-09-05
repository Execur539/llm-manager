/**
 * How a video is prepared before the model ever sees it.
 *
 * Every control here moves the token cost, which is the only reason any of them exist — a video
 * is the one attachment that can fill a context window on its own, and until now the two dials
 * that decided that were buried between the compaction strategy and the tool-call ceiling.
 *
 * The estimate is the point of the panel. Frames cost tokens by area and the trade between size
 * and rate is arithmetical, not a matter of taste, so the panel says what the current settings
 * would actually cost on a clip of a given length rather than describing the trade in words. It
 * is computed by the main process through the same planner that will run on the real upload, so
 * it cannot drift from what happens.
 */

import { useCallback, useEffect, useState } from 'react'
import type { AppSettings } from '@shared/types'
import { invoke } from '../lib/api'

interface Estimate {
  count: number
  width: number
  trackWidth: number
  stills: number
  costPerFrame: number
  costPerStill: number
  estimatedTokens: number
  effectiveFps: number
  contextLength: number
  modelLoaded: boolean
  serverTakesVideo: boolean
}

/** Lengths worth pricing: a clip, something from a phone, and a long screen capture. */
const SAMPLE_LENGTHS = [
  { label: '30s', seconds: 30 },
  { label: '2 min', seconds: 120 },
  { label: '10 min', seconds: 600 }
]

const DETAIL_LABELS: Record<AppSettings['video']['detail'], string> = {
  motion: 'Motion — many small frames',
  balanced: 'Balanced',
  detail: 'Detail — fewer, larger frames',
  high: 'High — 720p frames'
}

export default function VideoSettings({
  settings,
  patch
}: {
  settings: AppSettings
  patch: (p: Partial<AppSettings>) => void | Promise<void>
}): JSX.Element {
  const v = settings.video
  const [seconds, setSeconds] = useState(120)
  const [estimate, setEstimate] = useState<Estimate | null>(null)

  const set = useCallback(
    (change: Partial<AppSettings['video']>) => void patch({ video: { ...v, ...change } }),
    [patch, v]
  )

  /*
   * Re-priced whenever a control moves or the sample length changes.
   *
   * Depends on the settings object rather than on individual fields so a change made anywhere —
   * including from another window — is reflected without listing every key here.
   */
  useEffect(() => {
    let cancelled = false
    void (async () => {
      const next = await invoke<Estimate>('video:estimate', seconds).catch(() => null)
      if (!cancelled) setEstimate(next)
    })()
    return () => {
      cancelled = true
    }
  }, [v, seconds])

  const pct = estimate && estimate.contextLength > 0
    ? Math.round((estimate.estimatedTokens / estimate.contextLength) * 100)
    : 0

  return (
    <div className="video-settings" data-testid="video-settings">
      <Row label="Frame size" hint="the ceiling on how large one frame may be">
        <select
          value={v.detail}
          onChange={(e) => set({ detail: e.target.value as AppSettings['video']['detail'] })}
          data-testid="video-detail"
        >
          {(Object.keys(DETAIL_LABELS) as AppSettings['video']['detail'][]).map((k) => (
            <option key={k} value={k}>
              {DETAIL_LABELS[k]}
            </option>
          ))}
        </select>
      </Row>

      <Slider
        label="Frame rate ceiling"
        hint="sampling faster than this is refused however much budget is left"
        min={0.25}
        max={4}
        step={0.25}
        value={v.maxFps}
        format={(n) => `${n} fps`}
        onChange={(n) => set({ maxFps: n })}
        testId="video-max-fps"
      />

      <Slider
        label="Share of the context window"
        hint="the most one video may occupy, leaving room for the question and the answer"
        min={0.05}
        max={0.8}
        step={0.05}
        value={v.contextShare}
        format={(n) => `${Math.round(n * 100)}%`}
        onChange={(n) => set({ contextShare: n })}
        testId="video-share"
      />

      <Slider
        label="Video track size"
        hint={
          v.trackScale >= 1
            ? 'full size — the clip carries the detail itself and no stills are sent'
            : 'the clip is sent reduced, and full-size stills are sent alongside it'
        }
        min={0.25}
        max={1}
        step={0.05}
        value={v.trackScale}
        format={(n) => (n >= 1 ? 'full' : `${Math.round(n * 100)}%`)}
        onChange={(n) => set({ trackScale: n })}
        testId="video-track-scale"
      />

      <Slider
        label="Reserved for stills"
        hint="budget kept back for full-size frames at the cuts"
        min={0}
        max={0.6}
        step={0.05}
        value={v.stillShare}
        format={(n) => `${Math.round(n * 100)}%`}
        onChange={(n) => set({ stillShare: n })}
        // Nothing to reserve for: at full size the clip already carries the detail.
        disabled={v.trackScale >= 1}
        testId="video-still-share"
      />

      <Row label="Drop near-duplicate frames" hint="a still stretch collapses to one frame instead of many identical ones">
        <Toggle
          on={v.dropDuplicates}
          onChange={(on) => set({ dropDuplicates: on })}
          testId="video-drop-duplicates"
        />
      </Row>

      <Row label="Crop to the moving region" hint="black bars and a motionless background are not paid for">
        <Toggle on={v.cropStatic} onChange={(on) => set({ cropStatic: on })} testId="video-crop-static" />
      </Row>

      <div className="video-estimate" data-testid="video-estimate">
        <div className="video-estimate-head">
          <span>A</span>
          <div className="video-lengths">
            {SAMPLE_LENGTHS.map((s) => (
              <button
                key={s.seconds}
                className={seconds === s.seconds ? 'on' : ''}
                onClick={() => setSeconds(s.seconds)}
              >
                {s.label}
              </button>
            ))}
          </div>
          <span>video would cost</span>
        </div>

        {estimate ? (
          <>
            <div className="video-estimate-figure">
              <strong>{estimate.estimatedTokens.toLocaleString()}</strong> tokens
              {estimate.contextLength > 0 && <span className="faint"> · {pct}% of this model's window</span>}
            </div>
            <div className="video-estimate-detail">
              {estimate.count.toLocaleString()} frames at {estimate.effectiveFps.toFixed(2)} fps,{' '}
              {estimate.trackWidth}px wide ({estimate.costPerFrame} tokens each)
              {estimate.stills > 0 && (
                <>
                  , plus {estimate.stills} stills at {estimate.width}px ({estimate.costPerStill} each)
                </>
              )}
            </div>
            {/*
              * The figures are only as real as the model behind them. Without one loaded the
              * window is a placeholder, and saying so beats quoting a confident wrong number.
              */}
            {!estimate.modelLoaded && (
              <div className="video-estimate-note">
                No model is loaded, so this assumes a small default window. Load one for real figures.
              </div>
            )}
            {estimate.modelLoaded && !estimate.serverTakesVideo && (
              <div className="video-estimate-note">
                This server cannot take video directly, so frames are sent as separate images at
                double the cost above and without their timing.
              </div>
            )}
          </>
        ) : (
          <div className="video-estimate-detail faint">working it out…</div>
        )}
      </div>
    </div>
  )
}

function Row({
  label,
  hint,
  children
}: {
  label: string
  hint?: string
  children: React.ReactNode
}): JSX.Element {
  return (
    <div className="video-row">
      <div className="video-row-label">
        <span>{label}</span>
        {hint && <span className="video-row-hint">{hint}</span>}
      </div>
      <div className="video-row-control">{children}</div>
    </div>
  )
}

function Slider({
  label,
  hint,
  min,
  max,
  step,
  value,
  format,
  onChange,
  disabled,
  testId
}: {
  label: string
  hint?: string
  min: number
  max: number
  step: number
  value: number
  format: (n: number) => string
  onChange: (n: number) => void
  disabled?: boolean
  testId?: string
}): JSX.Element {
  return (
    <div className={`video-row${disabled ? ' is-disabled' : ''}`}>
      <div className="video-row-label">
        <span>{label}</span>
        {hint && <span className="video-row-hint">{hint}</span>}
      </div>
      <div className="video-row-control">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          /*
           * Committed on every move rather than on release. Each change is a settings write and a
           * re-price, both cheap, and a slider whose readout only catches up when you let go is
           * exactly the feedback loop this panel exists to close.
           */
          onChange={(e) => onChange(Number(e.target.value))}
          data-testid={testId}
        />
        <span className="video-row-value mono">{format(value)}</span>
      </div>
    </div>
  )
}

function Toggle({
  on,
  onChange,
  testId
}: {
  on: boolean
  onChange: (on: boolean) => void
  testId?: string
}): JSX.Element {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      className={`switch${on ? ' on' : ''}`}
      onClick={() => onChange(!on)}
      data-testid={testId}
    >
      <span className="switch-knob" />
    </button>
  )
}
