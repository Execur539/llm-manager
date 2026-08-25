import { useEffect, useState } from 'react'
import type { FitPlan, FitResult, ModelRecord } from '@shared/types'
import { fmtBytes, invoke } from '../lib/api'
import ConfirmDialog from '../components/ConfirmDialog'
import Icon from '../components/Icon'
import { Skeleton, Spinner } from '../components/Spinner'
import { toast } from '../lib/store'

interface ImportResult {
  imported: string[]
  skipped: string[]
  failed: { file: string; error: string }[]
  linked: number
}

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
        {busy ? <Spinner size={13} /> : null}
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
  const [pendingDelete, setPendingDelete] = useState<ModelRecord | null>(null)
  const [scanning, setScanning] = useState(false)

  const rescan = async (): Promise<void> => {
    setScanning(true)
    try {
      await onRefresh()
    } finally {
      setScanning(false)
    }
  }
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

  /**
   * Deleting a model erases tens of gigabytes that took a long download to obtain, and nothing
   * puts it back. The confirmation therefore names the file and the size, and — above a
   * threshold where re-downloading is a genuine cost — asks for the filename to be typed.
   */
  const remove = async (m: ModelRecord): Promise<void> => {
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
        <button onClick={() => void rescan()} disabled={scanning} data-testid="rescan">
          {scanning ? <Spinner size={13} /> : null}
          {scanning ? 'Scanning…' : 'Rescan'}
        </button>
        <button
          onClick={() => {
            void invoke<ImportResult>('library:import')
              .then(async (r) => {
                await onRefresh()
                if (!r) return
                const parts: string[] = []
                if (r.imported.length) {
                  // Say when nothing was duplicated — on a 20 GB model that is the difference
                  // between "instant" and "20 GB of disk gone".
                  parts.push(
                    `Imported ${r.imported.length} model${r.imported.length === 1 ? '' : 's'}` +
                      (r.linked === r.imported.length ? ' (linked, no extra disk used)' : '')
                  )
                }
                if (r.skipped.length) parts.push(`${r.skipped.length} already in the library`)
                if (r.failed.length) parts.push(`${r.failed.length} failed`)
                if (parts.length) {
                  toast(parts.join(' · '), r.failed.length ? 'error' : 'success')
                }
                for (const f of r.failed) toast(`${f.file}: ${f.error}`, 'error')
              })
              .catch((err: unknown) => toast(`Import failed: ${String(err)}`, 'error'))
          }}
          data-testid="import-gguf"
        >
          Import GGUF…
        </button>
        <button
          onClick={() => {
            void invoke<{ removed: number; bytes: number }>('library:clean-partials')
              .then(async (r) => {
                await onRefresh()
                // Silence here read as a broken button when there was nothing to clean.
                toast(
                  r && r.removed > 0
                    ? `Removed ${r.removed} partial download${r.removed === 1 ? '' : 's'}, freeing ${fmtBytes(r.bytes)}`
                    : 'No leftover partial downloads to clean up',
                  'success'
                )
              })
              .catch((err: unknown) => toast(`Clean-up failed: ${String(err)}`, 'error'))
          }}
          data-testid="clean-partials"
        >
          Clean partials
        </button>
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

            {/*
              * Delete used to be the only button on the card, which made the destructive action
              * the most prominent thing on screen while the primary one — loading the model —
              * had no affordance at all beyond "the card happens to be clickable".
              */}
            <div className="row model-actions" style={{ marginTop: 10 }}>
              <button
                className="primary"
                onClick={(e) => {
                  e.stopPropagation()
                  void select(m)
                }}
                title="Plan the fit for this model and choose how to load it"
                data-testid="plan-model"
              >
                {selected?.id === m.id ? 'Planning below…' : 'Load…'}
              </button>
              <button
                className="danger subtle"
                onClick={(e) => {
                  e.stopPropagation()
                  setPendingDelete(m)
                }}
                title={`Delete ${m.filename} from disk`}
                data-testid="delete-model"
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
              {fit.notes.length > 0 && (
                <div className="card fit-notes">
                  <div className="card-title">Why this plan</div>
                  <ul>
                    {fit.notes.map((n, i) => (
                      <li key={i}>
                        <Icon name="info" size={13} />
                        <span>{n}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

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

      <ConfirmDialog
        open={!!pendingDelete}
        danger
        title="Delete this model?"
        confirmLabel="Delete permanently"
        // Above 4 GB, re-downloading is a real cost — make it a deliberate, typed act.
        requirePhrase={pendingDelete && pendingDelete.bytes > 4 * 1024 ** 3 ? pendingDelete.filename : undefined}
        body={
          <>
            <p>
              <strong>{pendingDelete?.filename}</strong> will be erased from disk, freeing{' '}
              {fmtBytes(pendingDelete?.bytes)}. This cannot be undone and the file is not sent to the Recycle Bin.
            </p>
            {pendingDelete && pendingDelete.bytes > 4 * 1024 ** 3 && (
              <p className="subtitle">Type the filename to confirm.</p>
            )}
          </>
        }
        onCancel={() => setPendingDelete(null)}
        onConfirm={() => {
          const target = pendingDelete
          setPendingDelete(null)
          if (target) void remove(target)
        }}
      />
    </>
  )
}
