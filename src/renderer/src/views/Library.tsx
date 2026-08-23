import { useState } from 'react'
import type { FitPlan, FitResult, ModelRecord } from '@shared/types'

function fmt(n: number): string {
  const gb = n / 1024 ** 3
  return gb >= 1 ? `${gb.toFixed(2)} GB` : `${(n / 1024 ** 2).toFixed(0)} MB`
}

function CapBadges({ model }: { model: ModelRecord }): JSX.Element {
  return (
    <>
      {model.caps.vision && <span className="badge">vision</span>}
      {model.caps.audio && <span className="badge">audio</span>}
      {model.caps.nativeVideo ? (
        <span className="badge good">native video</span>
      ) : (
        model.caps.videoPossible && <span className="badge">video (frames)</span>
      )}
      {model.caps.tools && <span className="badge">tools</span>}
    </>
  )
}

/** Compatibility badge computed by the auto-fit engine before anything is loaded. */
function FitBadge({ fit }: { fit: FitResult | null }): JSX.Element {
  if (!fit) return <span className="badge">checking…</span>
  if (fit.chosen && !fit.chosen.spillsToHost) {
    return <span className="badge good">fits — {fit.chosen.contextLength.toLocaleString()} ctx</span>
  }
  if (fit.chosen?.spillsToHost) return <span className="badge warn">partial offload</span>
  if (fit.needsUserChoice) return <span className="badge warn">needs a choice</span>
  return <span className="badge bad">too large</span>
}

function PlanCard({ plan, onLoad }: { plan: FitPlan; onLoad: (p: FitPlan) => void }): JSX.Element {
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
        <dd>
          {plan.kvType} — {fmt(plan.kvBytes)}
        </dd>
        <dt>GPU layers</dt>
        <dd>
          {plan.gpuLayers} / {plan.totalLayers}
        </dd>
        <dt>Flash attention</dt>
        <dd>{plan.flashAttention ? 'on' : 'off'}</dd>
        <dt>Predicted VRAM</dt>
        <dd>{plan.predictedVramPerGpu.map(fmt).join(' + ') || 'n/a'}</dd>
        {plan.spillsToHost && (
          <>
            <dt>On system RAM</dt>
            <dd>{fmt(plan.predictedHostBytes)}</dd>
          </>
        )}
      </dl>
      <ul className="dim" style={{ fontSize: 12, paddingLeft: 16, marginBottom: 8 }}>
        {plan.rationale.map((r, i) => (
          <li key={i}>{r}</li>
        ))}
      </ul>
      <button className="primary" onClick={() => onLoad(plan)}>
        Load with this plan
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
  const [fit, setFit] = useState<FitResult | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const select = async (m: ModelRecord): Promise<void> => {
    setSelected(m)
    setFit(null)
    setError(null)
    const result = (await window.api.autofit.plan(m.id)) as FitResult | { error: string }
    if ('error' in result) setError(result.error)
    else setFit(result)
  }

  const load = async (plan: FitPlan): Promise<void> => {
    if (!selected) return
    setBusy(`Loading ${selected.filename}…`)
    setError(null)
    try {
      await window.api.model.load(selected.id, plan)
      await onLoaded()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(null)
    }
  }

  return (
    <>
      <h1>Models</h1>
      <p className="subtitle">
        {models.length} model{models.length === 1 ? '' : 's'} ·{' '}
        {fmt(models.reduce((a, m) => a + m.bytes, 0))} on disk
        <button style={{ marginLeft: 12 }} onClick={() => void onRefresh()}>
          Rescan
        </button>
      </p>

      {models.length === 0 && (
        <div className="card empty">
          No models found. Put .gguf files in the models folder beside the app, or use Find a model
          (not yet implemented — see BUILD_STATUS.md).
        </div>
      )}

      <div className="grid-2">
        {models.map((m) => (
          <div
            className="card"
            key={m.id}
            style={{ cursor: 'pointer', borderColor: selected?.id === m.id ? 'var(--accent-dim)' : undefined }}
            onClick={() => void select(m)}
          >
            <div className="card-title">
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{m.filename}</span>
            </div>
            <div className="row" style={{ flexWrap: 'wrap', marginBottom: 8 }}>
              <span className="badge">{fmt(m.bytes)}</span>
              {m.arch && <span className="badge">{m.arch.quant}</span>}
              {m.arch && <span className="badge">{m.arch.blockCount}L</span>}
              {m.arch && m.arch.contextLength > 0 && (
                <span className="badge">{(m.arch.contextLength / 1024).toFixed(0)}K trained</span>
              )}
              <CapBadges model={m} />
            </div>
            {m.error && <div className="badge bad">{m.error}</div>}
            {m.arch && (
              <div className="faint mono">
                {m.arch.architecture}
                {m.arch.headCountKv < m.arch.headCount ? ` · GQA ${m.arch.headCount}/${m.arch.headCountKv}` : ''}
              </div>
            )}
          </div>
        ))}
      </div>

      {selected && (
        <>
          <h1 style={{ marginTop: 24 }}>Fit for {selected.filename}</h1>
          {error && <div className="card badge bad">{error}</div>}
          {busy && <div className="card">{busy}</div>}

          {fit && (
            <>
              {fit.notes.map((n, i) => (
                <div className="card faint" key={i} style={{ padding: '8px 12px' }}>
                  {n}
                </div>
              ))}

              {fit.chosen && <PlanCard plan={fit.chosen} onLoad={(p) => void load(p)} />}

              {fit.needsUserChoice && (
                <>
                  <p className="subtitle">
                    The context target could not be met with everything on GPU. Nothing has been
                    changed silently — pick a tradeoff:
                  </p>
                  <div className="grid-2">
                    {fit.alternatives.map((p, i) => (
                      <PlanCard key={i} plan={p} onLoad={(pl) => void load(pl)} />
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
