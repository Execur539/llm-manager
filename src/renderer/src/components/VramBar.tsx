import type { FitPlan } from '@shared/types'
import { fmtBytes } from '../lib/api'

/**
 * What a plan puts on each GPU, stacked against the whole card.
 *
 * "9.6 GB + 10.2 GB" says how much, not whether it is a squeeze. Each bar's track is the card's
 * total memory, and it starts with what other programs already hold — the desktop, a browser — so
 * the empty space at the end is the headroom that actually remains, rather than a figure that
 * pretends the model has the card to itself. The colour steps up as a card passes 80% and 95%.
 */

/** Smallest width a non-empty segment is drawn at: a sliver under a pixel reads as nothing at all. */
const MIN_SEGMENT_PCT = 1.2

type Level = 'ok' | 'high' | 'critical'

function levelOf(fraction: number): Level {
  if (fraction >= 0.95) return 'critical'
  if (fraction >= 0.8) return 'high'
  return 'ok'
}

export function VramBars({ plan, compact = false }: { plan: FitPlan; compact?: boolean }): JSX.Element | null {
  const devices = plan.devices ?? []
  if (!devices.length || plan.gpuLayers <= 0) return null

  const rows = devices.map((device, d) => {
    const predicted = plan.predictedVramPerGpu[d] ?? 0
    // A plan stored before the split existed still has its per-card total.
    const seg = plan.vramSegments?.[d] ?? { weights: predicted, kv: 0, compute: 0 }
    const total = Math.max(1, device.totalVram)
    const others = device.measured && device.freeVram >= 0 ? Math.max(0, device.totalVram - device.freeVram) : 0
    const used = others + seg.weights + seg.kv + seg.compute
    return { device, seg, total, others, used, level: levelOf(used / total) }
  })

  const width = (bytes: number, total: number): string =>
    `${bytes > 0 ? Math.max(MIN_SEGMENT_PCT, (bytes / total) * 100) : 0}%`

  return (
    <div className={`vram-bars${compact ? ' compact' : ''}`}>
      {rows.map(({ device, seg, total, others, used, level }) => {
        const label = `${device.name}: ${fmtBytes(used)} of ${fmtBytes(total)} in use after loading`
        return (
          <div key={device.index} className="vram-row" title={compact ? label : undefined}>
            {!compact && (
              <div className="vram-row-head">
                <span className="vram-name">{device.name.replace(/^NVIDIA GeForce /, '')}</span>
                <span>
                  {fmtBytes(used)} of {fmtBytes(total)}
                </span>
              </div>
            )}
            <div className={`vram-bar ${level}`} role="img" aria-label={label}>
              {others > 0 && <span className="vram-seg others" style={{ width: width(others, total) }} />}
              <span className="vram-seg weights" style={{ width: width(seg.weights, total) }} />
              <span className="vram-seg kv" style={{ width: width(seg.kv, total) }} />
              <span className="vram-seg compute" style={{ width: width(seg.compute, total) }} />
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
          {/* The swatches show the resting colours, so what the warning colours mean is said here. */}
          <span className="vram-legend-note">amber past 80% of a card, red past 95%</span>
        </div>
      )}
    </div>
  )
}
