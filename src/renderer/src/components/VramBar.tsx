import type { FitPlan } from '@shared/types'
import { fmtBytes } from '../lib/api'

/**
 * What a plan puts on each GPU, stacked against the whole card.
 *
 * Each bar's track is the card's total memory. It starts with what other programs already hold, then
 * everything the load will take, including the parts outside llama.cpp's own buffers (the expert
 * cache, the vision projector, an MTP head, the CUDA runtime), which the old bars left out. The
 * empty space at the end is what the card keeps free.
 *
 * A full card is the point, not a warning: the planner fills what it was given. Red means the plan
 * leaves less free than the reserve, which only happens when a card filled up after planning.
 */

/** Smallest width a non-empty segment is drawn at: a sliver under a pixel reads as nothing at all. */
const MIN_SEGMENT_PCT = 1.2

export function VramBars({ plan, compact = false }: { plan: FitPlan; compact?: boolean }): JSX.Element | null {
  const devices = plan.devices ?? []
  if (!devices.length || plan.gpuLayers <= 0) return null

  const rows = devices.map((device, d) => {
    const predicted = plan.predictedVramPerGpu[d] ?? 0
    // A plan stored before the split existed still has its per-card total.
    const seg = plan.vramSegments?.[d] ?? { weights: predicted, kv: 0, compute: 0 }
    const total = Math.max(1, device.totalVram)
    const others = device.measured && device.freeVram >= 0 ? Math.max(0, device.totalVram - device.freeVram) : 0
    const parts = {
      weights: seg.weights,
      kv: seg.kv,
      compute: seg.compute,
      cache: seg.cache ?? 0,
      extras: seg.extras ?? 0
    }
    const used = others + parts.weights + parts.kv + parts.compute + parts.cache + parts.extras
    const free = total - used
    const reserve = plan.reservePerGpu?.[d] ?? 0
    return { device, parts, total, others, used, free, short: reserve > 0 && free < reserve * 0.9 }
  })

  const width = (bytes: number, total: number): string =>
    `${bytes > 0 ? Math.max(MIN_SEGMENT_PCT, (bytes / total) * 100) : 0}%`

  return (
    <div className={`vram-bars${compact ? ' compact' : ''}`}>
      {rows.map(({ device, parts, total, others, used, free, short }) => {
        const label = `${device.name}: ${fmtBytes(used)} of ${fmtBytes(total)} in use after loading, ${fmtBytes(Math.max(0, free))} free`
        return (
          <div key={device.index} className="vram-row" title={compact ? label : undefined}>
            {!compact && (
              <div className="vram-row-head">
                <span className="vram-name">{device.name.replace(/^NVIDIA GeForce /, '')}</span>
                <span>
                  {fmtBytes(used)} of {fmtBytes(total)} · {fmtBytes(Math.max(0, free))} free
                </span>
              </div>
            )}
            <div className={`vram-bar ${short ? 'critical' : 'ok'}`} role="img" aria-label={label}>
              {others > 0 && <span className="vram-seg others" style={{ width: width(others, total) }} />}
              <span className="vram-seg weights" style={{ width: width(parts.weights, total) }} />
              <span className="vram-seg kv" style={{ width: width(parts.kv, total) }} />
              <span className="vram-seg compute" style={{ width: width(parts.compute, total) }} />
              {parts.cache > 0 && <span className="vram-seg cache" style={{ width: width(parts.cache, total) }} />}
              {parts.extras > 0 && <span className="vram-seg extras" style={{ width: width(parts.extras, total) }} />}
            </div>
          </div>
        )
      })}
      {!compact && (
        <div className="vram-legend">
          <span>
            <i className="vram-seg others" />
            other programs
          </span>
          <span>
            <i className="vram-seg weights" />
            weights
          </span>
          <span>
            <i className="vram-seg kv" />
            KV cache
          </span>
          <span>
            <i className="vram-seg compute" />
            working memory
          </span>
          {rows.some((r) => r.parts.cache > 0) && (
            <span>
              <i className="vram-seg cache" />
              expert cache
            </span>
          )}
          {rows.some((r) => r.parts.extras > 0) && (
            <span>
              <i className="vram-seg extras" />
              runtime, vision, MTP head
            </span>
          )}
          {rows.some((r) => r.short) && (
            <span className="vram-legend-note">Red: that card would keep less free than your reserve</span>
          )}
        </div>
      )}
    </div>
  )
}
