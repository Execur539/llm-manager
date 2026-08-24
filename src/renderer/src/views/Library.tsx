import { useEffect, useState } from 'react'
import type { FitPlan, FitResult, ModelRecord } from '@shared/types'
import { fmtBytes, invoke } from '../lib/api'

function CapBadges({ model }: { model: ModelRecord }): JSX.Element {
  return (
    <>
      {model.caps.vision && <span className="badge">vision</span>}
      {model.caps.audio && <span className="badge">audio</span>}
      {model.caps.nativeVideo ? (
        <span className="badge good">native video</span>
      ) : (
        model.caps.videoPossible && <span className="badge">video via frames</span>
      )}
      {model.caps.tools && <span className="badge">tools</span>}
    </>
  )
}

function FitBadge({ fit }: { fit: FitResult | { error: string } | null }): JSX.Element {
  if (!fit) return <span className="badge">checking…</span>
  if ('error' in fit) return <span className="badge bad">unreadable</span>
  if (fit.chosen && !fit.chosen.spillsToHost) {
    return <span className="badge good">fits · {(fit.chosen.contextLength / 1024).toFixed(0)}K ctx</span>
  }
  if (fit.chosen?.spillsToHost) return <span className="badge warn">partial offload</span>
  if (fit.needsUserChoice) return <span className="badge warn">tradeoff needed</span>
  return <span className="badge bad">too large</span>
}

function PlanCard({ plan, onLoad, busy }: { plan: FitPlan; onLoad: (p: FitPlan) => void; busy: boolean }): JSX.Element {
  return (
    <div className="card">
      <div className="card-title">
        {plan.label}
        <span className="badge">{plan.speedScore}/100 speed</span>
      </div>
      <dl className="kv">
        <dt>Context</dt>
        <dd>{plan.contextLength.toLocaleString()} tokens</dd>
        <dt>KV cache</dt>
        <dd>{plan.kvType} · {fmtBytes(plan.kvBytes)}</dd>
        <dt>GPU layers</dt>
        <dd>{plan.gpuLayers} / {plan.totalLayers}</dd>
        <dt>Flash attention</dt>
        <dd>{plan.flashAttention ? 'on' : 'off'}</dd>
        <dt>Predicted VRAM</dt>
        <dd>{plan.predictedVramPerGpu.map((v) => fmtBytes(v)).join(' + ') || '—'}</dd>
        {plan.tensorSplit.length > 1 && (
          <>
            <dt>Split</dt>
            <dd>{plan.tensorSplit.map((s) => `${(s * 100).toFixed(0)}%`).join(' / ')}</dd>
          </>
        )}
        {plan.spillsToHost && (
          <>
            <dt>On system RAM</dt>
            <dd>{fmtBytes(plan.predictedHostBytes)}</dd>
          </>
        )}
      </dl>
      <ul className="rationale">
        {plan.rationale.map((r, i) => (
          <li key={i}>{r}</li>
        ))}
      </ul>
      <button className="primary" disabled={busy} onClick={() => onLoad(plan)}>
        {busy ? 'Loading…' : 'Load with this plan'}
      </button>
    </div>
  )
}

export default function Library({
  models,
  onRefresh,
  onLoaded
}: {
  models: ModelRecord[]
  onRefresh: () => Promise<void>
  onLoaded: () => Promise<void>
}): JSX.Element {
  const [selected, setSelected] = useState<ModelRecord | null>(null)
  const [fit, setFit] = useState<FitResult | { error: string } | null>(null)
  const [fits, setFits] = useState<Record<string, FitResult | { error: string }>>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [disk, setDisk] = useState<{ totalBytes: number; freeBytes: number; partialBytes: number } | null>(null)
  const [filter, setFilter] = useState('')
  const [showFavourites, setShowFavourites] = useState(false)

  useEffect(() => {
    void invoke<typeof disk>('library:disk').then(setDisk).catch(() => undefined)
  }, [models.length])

  // Compute compatibility badges up front — the plan calls for them before loading, not after.
  useEffect(() => {
    let cancelled = false
    void (async () => {
      for (const m of models) {
        if (cancelled || fits[m.id]) continue
        try {
          const result = await invoke<FitResult | { error: string }>('autofit:plan', m.id)
          if (!cancelled) setFits((prev) => ({ ...prev, [m.id]: result }))
        } catch {
          /* skip this one */
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [models])

  const select = async (m: ModelRecord): Promise<void> => {
    setSelected(m)
    setError(null)
    setFit(fits[m.id] ?? null)
    const result = await invoke<FitResult | { error: string }>('autofit:plan', m.id)
    setFit(result)
    setFits((prev) => ({ ...prev, [m.id]: result }))
  }

  const load = async (plan: FitPlan): Promise<void> => {
    if (!selected) return
    setBusy(true)
    setError(null)
    try {
      await invoke('model:load', selected.id, plan)
      await onLoaded()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const remove = async (m: ModelRecord): Promise<void> => {
    if (!confirm(`Delete ${m.filename}? This removes ${fmtBytes(m.bytes)} from disk permanently.`)) return
    try {
      await invoke('library:delete-model', m.id)
      if (selected?.id === m.id) setSelected(null)
      await onRefresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const visible = models
    .filter((m) => !showFavourites || m.favourite)
    .filter((m) => !filter || m.filename.toLowerCase().includes(filter.toLowerCase()) || m.tags.some((t) => t.includes(filter)))

  return (
    <>
      <h1>My models</h1>
      <p className="subtitle">
        {models.length} model{models.length === 1 ? '' : 's'} · {fmtBytes(models.reduce((a, m) => a + m.bytes, 0))} on disk
        {disk && disk.freeBytes >= 0 && ` · ${fmtBytes(disk.freeBytes)} free`}
        {disk && disk.partialBytes > 0 && ` · ${fmtBytes(disk.partialBytes)} in partial downloads`}
      </p>

      <div className="row" style={{ marginBottom: 14, flexWrap: 'wrap' }}>
        <input
          type="text"
          placeholder="Filter by name or tag…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          style={{ flex: 1, minWidth: 200 }}
        />
        <button className={showFavourites ? 'primary' : ''} onClick={() => setShowFavourites((v) => !v)}>
          Favourites
        </button>
        <button onClick={() => void onRefresh()}>Rescan</button>
        <button onClick={() => void invoke('library:import').then(() => onRefresh())}>Import GGUF…</button>
        <button onClick={() => void invoke('library:clean-partials').then(() => onRefresh())}>Clean partials</button>
      </div>

      {error && (
        <div className="card" style={{ borderColor: '#5c2626' }}>
          <span className="badge bad">error</span> {error}
        </div>
      )}

      {models.length === 0 && (
        <div className="card empty">
          No models yet. Use <strong>Find a model</strong> to search HuggingFace, or drop .gguf files into the
          models folder beside the app.
        </div>
      )}

      <div className="grid-2">
        {visible.map((m) => (
          <div
            className={`card model-card ${selected?.id === m.id ? 'selected' : ''}`}
            key={m.id}
            onClick={() => void select(m)}
          >
            <div className="card-title">
              <span className="truncate" title={m.filename}>{m.filename}</span>
              <button
                className="icon"
                title={m.favourite ? 'Unfavourite' : 'Favourite'}
                onClick={(e) => {
                  e.stopPropagation()
                  void invoke('library:set-favourite', m.id, !m.favourite).then(onRefresh)
                }}
              >
                {m.favourite ? '★' : '☆'}
              </button>
            </div>

            <div className="row" style={{ flexWrap: 'wrap', marginBottom: 8 }}>
              <FitBadge fit={fits[m.id] ?? null} />
              <span className="badge">{fmtBytes(m.bytes)}</span>
              {(m.quantLabel || m.arch) && (
                <span
                  className="badge"
                  title={
                    m.mixedQuant && m.arch
                      ? `Filename says ${m.quantLabel}; most tensors are ${m.arch.quant} (mixed-precision quant)`
                      : undefined
                  }
                >
                  {m.quantLabel ?? m.arch?.quant}
                  {m.mixedQuant && ' *'}
                </span>
              )}
              {m.arch && m.arch.contextLength > 0 && (
                <span className="badge">{(m.arch.contextLength / 1024).toFixed(0)}K trained</span>
              )}
              <CapBadges model={m} />
            </div>

            {m.error && <div className="badge bad">{m.error}</div>}

            {m.arch && (
              <div className="faint mono" style={{ fontSize: 11 }}>
                {m.arch.architecture} · {m.arch.blockCount} layers
                {m.arch.headCountKv < m.arch.headCount && ` · GQA ${m.arch.headCount}/${m.arch.headCountKv}`}
                {m.arch.expertCount > 0 && ` · MoE ${m.arch.expertCount}`}
              </div>
            )}

            <div className="row" style={{ marginTop: 10 }}>
              <button
                className="danger"
                onClick={(e) => {
                  e.stopPropagation()
                  void remove(m)
                }}
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>

      {selected && (
        <>
          <h1 style={{ marginTop: 26 }}>Fit for {selected.filename}</h1>

          {fit && 'error' in fit && (
            <div className="card">
              <span className="badge bad">cannot plan</span> {fit.error}
            </div>
          )}

          {fit && !('error' in fit) && (
            <>
              {fit.notes.map((n, i) => (
                <div className="card note" key={i}>{n}</div>
              ))}

              {fit.chosen && <PlanCard plan={fit.chosen} onLoad={(p) => void load(p)} busy={busy} />}

              {fit.needsUserChoice && (
                <>
                  <p className="subtitle">
                    The context target could not be met with every layer on GPU. Nothing was changed silently —
                    choose which tradeoff you want:
                  </p>
                  <div className="grid-2">
                    {fit.alternatives.map((p, i) => (
                      <PlanCard key={i} plan={p} onLoad={(pl) => void load(pl)} busy={busy} />
                    ))}
                  </div>
                </>
              )}
            </>
          )}
        </>
      )}
    </>
  )
}
