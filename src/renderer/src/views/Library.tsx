import { useEffect, useRef, useState } from 'react'
import type { FitPlan, FitResult, ModelRecord } from '@shared/types'
import { displayModelName, fmtBytes, invoke } from '../lib/api'
import ConfirmDialog from '../components/ConfirmDialog'
import Icon from '../components/Icon'
import { Skeleton, Spinner } from '../components/Spinner'
import { VramBars } from '../components/VramBar'
import { toast } from '../lib/store'
import { Toolbar } from '../lib/shell'

interface ImportResult {
  imported: string[]
  skipped: string[]
  failed: { file: string; error: string }[]
  linked: number
}

/** A context length as people say it: 500K, 1M. */
function fmtContext(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_048_576).toFixed(n % 1_048_576 ? 1 : 0)}M`
  return `${Math.round(n / 1000)}K`
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
      {(model.arch?.mtpLayers ?? 0) > 0 ? (
        <span className="badge good" title="Built-in multi-token prediction: drafts ahead of itself for faster generation">
          MTP
        </span>
      ) : (
        model.caps.mtpModule && (
          <span className="badge good" title={`Multi-token prediction from ${model.caps.mtpModule.path}: drafts ahead of itself for faster generation`}>
            MTP
          </span>
        )
      )}
      {(model.parts?.length ?? 0) > 1 && <span className="badge">{model.parts?.length} parts</span>}
    </>
  )
}

function FitBadge({ fit }: { fit: FitResult | { error: string } | null }): JSX.Element {
  if (!fit) return <span className="badge fit-badge checking">Checking…</span>
  if ('error' in fit) return <span className="badge bad fit-badge">Unreadable</span>
  if (fit.chosen && !fit.chosen.spillsToHost) {
    return <span className="badge good fit-badge">Fits · {fmtContext(fit.chosen.contextLength)} context</span>
  }
  // Every block on the GPU with some routed experts in RAM is how an MoE model normally runs, not a degraded one.
  if (fit.chosen && fit.chosen.gpuLayers >= fit.chosen.totalLayers && (fit.chosen.cpuMoeLayers ?? 0) > 0) {
    return (
      <span className="badge good fit-badge" title="Attention and shared weights on the GPU; some routed experts in system RAM">
        Fits · {fmtContext(fit.chosen.contextLength)} context, experts in RAM
      </span>
    )
  }
  if (fit.chosen?.spillsToHost) return <span className="badge warn fit-badge">Partial offload</span>
  if (fit.needsUserChoice) return <span className="badge warn fit-badge">Tradeoff needed</span>
  return <span className="badge bad fit-badge">Too large</span>
}

/** The options side by side, when there is more than one way to load the model. */
function PlanPicker({
  plans,
  index,
  onPick
}: {
  plans: FitPlan[]
  index: number
  onPick: (i: number) => void
}): JSX.Element {
  return (
    <div className="plan-picker" role="radiogroup" aria-label="Ways to load this model">
      {plans.map((p, i) => (
        <button
          key={i}
          type="button"
          role="radio"
          aria-checked={i === index}
          className={`plan-option${i === index ? ' selected' : ''}`}
          onClick={() => onPick(i)}
          data-testid="plan-option"
        >
          <span className="plan-option-label">{p.label}</span>
          <span className="plan-option-figure">{fmtContext(p.contextLength)}</span>
          <span className="plan-option-sub">
            {p.tokensPerSecond ? `≈ ${Math.round(p.tokensPerSecond)} tok/s` : `${p.speedScore}/100 speed`}
          </span>
        </button>
      ))}
    </div>
  )
}

/** One plan: what it gives you in large figures, where its memory goes, then the detail. */
function PlanDetail({ plan }: { plan: FitPlan }): JSX.Element {
  const cache = plan.cacheVramPerGpu?.reduce((a, b) => a + b, 0) ?? 0
  const speedTitle =
    plan.speedSource === 'measured'
      ? 'Measured: the median of replies this model wrote on this machine with this plan'
      : plan.speedSource === 'calibrated'
        ? "Estimated, then scaled by how this model's other plans actually ran on this machine"
        : 'Estimated from where the weights sit; it becomes measured once the model has written a few replies'
  return (
    <div className="plan-detail">
      <div className="plan-figures">
        <div className="stat">
          <div className="stat-label">Context</div>
          <div className="stat-value">
            {fmtContext(plan.contextLength)}
            <span className="stat-unit">tokens</span>
          </div>
          <div className="stat-sub">{plan.contextLength.toLocaleString()}</div>
        </div>
        <div className="stat" title={plan.tokensPerSecond ? speedTitle : undefined}>
          <div className="stat-label">Speed</div>
          <div className="stat-value">
            {plan.tokensPerSecond ? `${plan.speedSource === 'measured' ? '' : '≈ '}${Math.round(plan.tokensPerSecond)}` : plan.speedScore}
            <span className="stat-unit">{plan.tokensPerSecond ? 'tok/s' : '/100'}</span>
          </div>
          <div className="stat-sub">
            {plan.speedSource === 'measured' ? 'measured' : plan.speedSource === 'calibrated' ? 'from your runs' : 'estimate'}
          </div>
        </div>
        <div className="stat">
          <div className="stat-label">KV cache</div>
          <div className="stat-value small">
            {plan.kvTypeV && plan.kvTypeV !== plan.kvType ? `${plan.kvType} / ${plan.kvTypeV}` : plan.kvType}
          </div>
          <div className="stat-sub">{fmtBytes(plan.kvBytes)}</div>
        </div>
        {(plan.expertCacheSlots ?? 0) > 0 ? (
          <div className="stat">
            <div className="stat-label">Expert cache</div>
            <div className="stat-value">
              {plan.expertCacheSlots}
              <span className="stat-unit">slots</span>
            </div>
            <div className="stat-sub">{fmtBytes(cache)} on the GPUs</div>
          </div>
        ) : (
          <div className="stat">
            <div className="stat-label">Layers on GPU</div>
            <div className="stat-value">
              {plan.gpuLayers}
              <span className="stat-unit">of {plan.totalLayers}</span>
            </div>
          </div>
        )}
      </div>

      <VramBars plan={plan} />

      <details className="plan-more">
        <summary>
          Details
          <Icon name="chevron" size={13} />
        </summary>
        <dl className="kv plan-facts">
          <dt>GPU layers</dt>
          <dd>
            {plan.gpuLayers} of {plan.totalLayers}
          </dd>
          {(plan.cpuMoeLayers ?? 0) > 0 && (
            <>
              <dt>Experts in RAM</dt>
              <dd>first {plan.cpuMoeLayers} layers</dd>
            </>
          )}
          {plan.tensorSplit.length > 1 && (
            <>
              <dt>Split</dt>
              <dd>
                {plan.layerSplit && plan.layerSplit.length > 1
                  ? plan.layerSplit.join(' + ') + ' layers'
                  : plan.tensorSplit.map((x) => `${(x * 100).toFixed(0)}%`).join(' / ')}
              </dd>
            </>
          )}
          <dt>Predicted VRAM</dt>
          <dd>
            {plan.predictedVramPerGpu.map((v) => fmtBytes(v)).join(' + ') || '—'}
            <span className="faint">{plan.measured ? 'confirmed by llama.cpp' : plan.llamaVramPerGpu ? 'estimate' : ''}</span>
          </dd>
          {plan.spillsToHost && (
            <>
              <dt>In system RAM</dt>
              <dd>{fmtBytes(plan.predictedHostBytes)}</dd>
            </>
          )}
          <dt>Flash attention</dt>
          <dd>{plan.flashAttention ? 'on' : 'off'}</dd>
        </dl>
        <ul className="rationale">
          {plan.rationale.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
      </details>
    </div>
  )
}

interface PlacementOverrides {
  contextLength?: number
  gpuLayers?: number
  cpuMoeLayers?: number
  kvType?: 'f16' | 'q8_0' | 'q4_0'
  kvTypeV?: 'f16' | 'q8_0' | 'q4_0'
  overrideTensors?: string
}

/**
 * The manual escape hatch for placement.
 *
 * The planner decides all of this on its own and should be right; this is for when someone knows
 * better for their machine, or wants to try a placement it does not model. Empty means "let the
 * planner decide", and anything set is honoured exactly rather than adjusted.
 */
function AdvancedPlacement({
  model,
  value,
  onApply,
  busy
}: {
  model: ModelRecord
  value: PlacementOverrides
  onApply: (o: PlacementOverrides) => void
  busy: boolean
}): JSX.Element {
  const [draft, setDraft] = useState<PlacementOverrides>(value)
  useEffect(() => setDraft(value), [value])
  const num = (s: string): number | undefined => {
    const n = Number(s)
    return s.trim() === '' || !Number.isFinite(n) ? undefined : Math.max(0, Math.round(n))
  }
  const moe = (model.arch?.expertCount ?? 0) > 0
  const active = Object.values(value).some((v) => v !== undefined && v !== '')
  const kvOptions = (
    <>
      <option value="f16">f16</option>
      <option value="q8_0">q8_0</option>
      <option value="q4_0">q4_0</option>
    </>
  )
  return (
    <details className="card advanced-placement" open={active}>
      <summary>
        Advanced placement {active && <span className="badge warn">overrides on</span>}
      </summary>
      <p className="faint" style={{ fontSize: 11, margin: '8px 0' }}>
        Leave a field empty to let the planner decide. Anything set is passed to llama.cpp exactly as given.
      </p>
      <dl className="kv kv-wide">
        <dt>Context</dt>
        <dd>
          <input
            type="number"
            min={512}
            placeholder="auto"
            value={draft.contextLength ?? ''}
            onChange={(e) => setDraft({ ...draft, contextLength: num(e.target.value) })}
          />
        </dd>
        <dt>GPU layers</dt>
        <dd>
          <input
            type="number"
            min={0}
            max={model.arch?.blockCount}
            placeholder="auto"
            value={draft.gpuLayers ?? ''}
            onChange={(e) => setDraft({ ...draft, gpuLayers: num(e.target.value) })}
          />
          <span className="faint">blocks on the GPU, counted from the last</span>
        </dd>
        {moe && (
          <>
            <dt>Experts in RAM</dt>
            <dd>
              <input
                type="number"
                min={0}
                max={model.arch?.blockCount}
                placeholder="auto"
                value={draft.cpuMoeLayers ?? ''}
                onChange={(e) => setDraft({ ...draft, cpuMoeLayers: num(e.target.value) })}
                data-testid="override-cpu-moe"
              />
              <span className="faint">--n-cpu-moe: routed experts of the first N layers stay in system RAM</span>
            </dd>
          </>
        )}
        <dt>Key cache</dt>
        <dd>
          <select
            className="kv-select"
            value={draft.kvType ?? ''}
            onChange={(e) => {
              const k = (e.target.value || undefined) as PlacementOverrides['kvType']
              setDraft({ ...draft, kvType: k, kvTypeV: k ? draft.kvTypeV : undefined })
            }}
          >
            <option value="">auto</option>
            {kvOptions}
          </select>
          <span className="faint">--cache-type-k</span>
        </dd>
        <dt>Value cache</dt>
        <dd>
          <select
            className="kv-select"
            disabled={!draft.kvType}
            value={draft.kvTypeV ?? ''}
            onChange={(e) =>
              setDraft({ ...draft, kvTypeV: (e.target.value || undefined) as PlacementOverrides['kvTypeV'] })
            }
          >
            <option value="">same as keys</option>
            {kvOptions}
          </select>
          <span className="faint">--cache-type-v</span>
        </dd>
        <dt>Tensor overrides</dt>
        <dd>
          <input
            type="text"
            placeholder="e.g. blk\.(4[0-7])\.ffn_.*_exps=CPU"
            value={draft.overrideTensors ?? ''}
            onChange={(e) => setDraft({ ...draft, overrideTensors: e.target.value || undefined })}
            data-testid="override-tensors"
          />
          <span className="faint">--override-tensor, not counted in the memory prediction</span>
        </dd>
      </dl>
      <div className="row" style={{ marginTop: 10 }}>
        <button className="primary" disabled={busy} onClick={() => onApply(draft)} data-testid="override-apply">
          Re-plan
        </button>
        <button disabled={busy || !active} onClick={() => onApply({})}>
          Reset to automatic
        </button>
      </div>
    </details>
  )
}

export default function Library({
  models,
  onRefresh,
  onLoaded,
  detection = null
}: {
  models: ModelRecord[]
  onRefresh: () => Promise<void>
  onLoaded: () => Promise<void>
  /** Whether GPU detection has answered for certain; badges planned before it did are planned again. */
  detection?: 'measured' | 'provisional' | null
}): JSX.Element {
  const [selected, setSelected] = useState<ModelRecord | null>(null)
  const [fit, setFit] = useState<FitResult | { error: string } | null>(null)
  const [fits, setFits] = useState<Record<string, FitResult | { error: string }>>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pendingDelete, setPendingDelete] = useState<ModelRecord | null>(null)
  const [scanning, setScanning] = useState(false)
  const [overrides, setOverrides] = useState<PlacementOverrides>({})
  /** Which of the offered plans is picked in the sheet; the chosen one, until you pick another. */
  const [planIndex, setPlanIndex] = useState(0)
  useEffect(() => setPlanIndex(0), [fit])
  const plans: FitPlan[] =
    fit && !('error' in fit) ? [fit.chosen, ...fit.alternatives].filter((x): x is FitPlan => !!x) : []
  const current = plans[Math.min(planIndex, Math.max(0, plans.length - 1))] ?? null

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

  /*
   * Compute compatibility badges up front — the plan calls for them before loading, not after.
   *
   * Each badge is computed once and kept, so one planned while GPU detection was still a guess is
   * planned again when detection answers. Otherwise a library opened in the first seconds after
   * launch goes on showing plans that leave out GPUs found a moment later.
   */
  const plannedUnder = useRef<Record<string, 'measured' | 'provisional' | null>>({})
  useEffect(() => {
    let cancelled = false
    void (async () => {
      for (const m of models) {
        const stale = plannedUnder.current[m.id] === 'provisional' && detection !== 'provisional'
        if (cancelled || (fits[m.id] && !stale)) continue
        try {
          // The estimate is enough for a badge; confirming every model would cost seconds each.
          const result = await invoke<FitResult | { error: string }>('autofit:plan', m.id, undefined, false)
          if (cancelled) continue
          plannedUnder.current[m.id] = detection
          setFits((prev) => ({ ...prev, [m.id]: result }))
        } catch {
          /* skip this one */
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [models, detection])

  const select = async (m: ModelRecord): Promise<void> => {
    setSelected(m)
    setError(null)
    setOverrides({})
    setFit(fits[m.id] ?? null)
    // The dialog shows what will load, so its plans are confirmed with llama.cpp's own dry run.
    const result = await invoke<FitResult | { error: string }>('autofit:plan', m.id, undefined, true)
    setFit(result)
    setFits((prev) => ({ ...prev, [m.id]: result }))
  }

  // A plan with overrides is for this dialog only; the library badge stays the automatic one.
  const replan = async (o: PlacementOverrides): Promise<void> => {
    if (!selected) return
    setOverrides(o)
    setFit(null)
    setError(null)
    const clean = Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined && v !== ''))
    setFit(await invoke<FitResult | { error: string }>('autofit:plan', selected.id, clean, true))
  }

  /*
   * Escape closes the plan, except while a load is in flight.
   *
   * Dismissing it mid-load would leave the model loading with the surface that reports it gone,
   * and no way back to the error if it failed.
   */
  useEffect(() => {
    if (!selected) return
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape' && !busy) {
        e.preventDefault()
        setSelected(null)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selected, busy])

  const load = async (plan: FitPlan): Promise<void> => {
    if (!selected) return
    setBusy(true)
    setError(null)
    try {
      await invoke('model:load', selected.id, plan)
      await onLoaded()
      // The decision is made and acted on; leaving it open would hold a dialog over the result.
      setSelected(null)
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

  const importGguf = (): void => {
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
  }
  const addFolder = (): void => {
    void invoke<{ added: string | null; coveredBy?: string; folders: string[] }>('library:add-folder')
      .then(async (r) => {
        if (!r) return
        if (r.added) {
          await onRefresh()
          toast(`Models in ${r.added} are now listed here.`, 'success')
        } else if (r.coveredBy) {
          toast(`That folder is already covered by ${r.coveredBy}.`, 'info')
        }
      })
      .catch((err: unknown) => toast(`Could not add the folder: ${String(err)}`, 'error'))
  }
  const cleanPartials = (): void => {
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
  }

  const favourites = models.filter((m) => m.favourite).length

  return (
    <>
      {/* Bringing models in is what this page adds to; the two ways to do it sit with its title. */}
      <Toolbar title="My models">
        <button onClick={importGguf} data-testid="import-gguf" title="Copy or link .gguf files into the library">
          <Icon name="plus" size={15} />
          Import GGUF…
        </button>
        <button
          title="List the models in another folder here, without moving or copying them"
          onClick={addFolder}
          data-testid="library-add-folder"
        >
          <Icon name="folder" size={15} />
          Add folder…
        </button>
      </Toolbar>

      <div className="library-summary">
        <span>
          {models.length} model{models.length === 1 ? '' : 's'}
        </span>
        <span>{fmtBytes(models.reduce((a, m) => a + m.bytes, 0))} on disk</span>
        {disk && disk.freeBytes >= 0 && <span>{fmtBytes(disk.freeBytes)} free</span>}
        {disk && disk.partialBytes > 0 && <span>{fmtBytes(disk.partialBytes)} in partial downloads</span>}
      </div>

      <div className="library-controls">
        <label className="search-field">
          <Icon name="search" size={15} />
          <input
            type="text"
            placeholder="Filter by name or tag…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            aria-label="Filter models"
          />
        </label>
        <div className="segmented" role="group" aria-label="Show">
          <button aria-pressed={!showFavourites} onClick={() => setShowFavourites(false)}>
            All
          </button>
          <button aria-pressed={showFavourites} onClick={() => setShowFavourites(true)}>
            Favourites{favourites > 0 && <span className="segmented-count">{favourites}</span>}
          </button>
        </div>
        <span className="spacer" />
        <button className="ghost" onClick={() => void rescan()} disabled={scanning} data-testid="rescan">
          {scanning ? <Spinner size={13} /> : null}
          {scanning ? 'Scanning…' : 'Rescan'}
        </button>
        <button className="ghost" onClick={cleanPartials} data-testid="clean-partials" title="Delete downloads that never finished">
          Clean partials
        </button>
      </div>

      {/*
        * Suppressed while the plan dialog is up, where the same error is shown instead.
        *
        * A failed load leaves the dialog open on purpose — you need to see which plan failed and
        * pick another. Rendered only here, the reason why would sit on the page behind the
        * backdrop, so the load would appear to do nothing at all.
        */}
      {error && !selected && (
        <div className="card error-card">
          <span className="badge bad">error</span> {error}
        </div>
      )}

      {models.length === 0 && (
        <div className="card empty">
          No models yet. Use <strong>Find a model</strong> to search Hugging Face, or drop .gguf files into the
          models folder beside the app.
        </div>
      )}
      {models.length > 0 && visible.length === 0 && (
        <div className="card empty small">{showFavourites && !filter ? 'No favourites yet. Star a model to keep it here.' : 'No model matches that filter.'}</div>
      )}

      {/*
        * One list, a row per model: the name and what it is on the left, how it fits and what to do
        * with it on the right. The fit is the thing you came to read, so it leads the second line.
        */}
      {visible.length > 0 && (
        <div className="model-list">
          {visible.map((m) => {
            const f = fits[m.id] ?? null
            const plan = f && !('error' in f) ? (f.chosen ?? f.alternatives[0] ?? null) : null
            const quant = m.quantLabel ?? m.arch?.quant
            return (
              <div
                className={`model-card${selected?.id === m.id ? ' selected' : ''}${m.error ? ' broken' : ''}`}
                key={m.id}
                onClick={() => void select(m)}
              >
                <div className="model-main">
                  <div className="model-title">
                    <span className="model-name truncate" title={m.filename}>
                      {displayModelName(m.filename)}
                    </span>
                    <button
                      className={`model-fav${m.favourite ? ' on' : ''}`}
                      title={m.favourite ? 'Unfavourite' : 'Favourite'}
                      aria-label={m.favourite ? 'Unfavourite' : 'Favourite'}
                      aria-pressed={m.favourite}
                      onClick={(e) => {
                        e.stopPropagation()
                        void invoke('library:set-favourite', m.id, !m.favourite).then(onRefresh)
                      }}
                    >
                      <Icon name={m.favourite ? 'star-filled' : 'star'} size={14} />
                    </button>
                  </div>
                  <div className="model-meta">
                    <FitBadge fit={f} />
                    <span>{fmtBytes(m.bytes)}</span>
                    {quant && (
                      <span
                        title={
                          m.mixedQuant && m.arch
                            ? `Filename says ${m.quantLabel}; most tensors are ${m.arch.quant} (mixed-precision quant)`
                            : undefined
                        }
                      >
                        {quant}
                        {m.mixedQuant && ' *'}
                      </span>
                    )}
                    {m.arch && m.arch.contextLength > 0 && <span>{(m.arch.contextLength / 1024).toFixed(0)}K trained</span>}
                    {m.arch && (
                      <span className="model-arch">
                        {m.arch.architecture} · {m.arch.blockCount} layers
                        {m.arch.expertCount > 0 && ` · ${m.arch.expertCount} experts`}
                      </span>
                    )}
                  </div>
                  <div className="model-tags">
                    <CapBadges model={m} />
                  </div>
                  {m.error && <div className="badge bad model-error">{m.error}</div>}
                </div>

                {/* The plan this row would load with, drawn per card so a squeeze shows before opening it. */}
                <div className="model-fit">{plan ? <VramBars plan={plan} compact /> : null}</div>

                {/*
                  * Delete used to be the only button on the card, which made the destructive action
                  * the most prominent thing on screen while the primary one, loading the model, had no
                  * affordance at all beyond "the card happens to be clickable".
                  */}
                <div className="model-actions">
                  <button
                    className="primary"
                    onClick={(e) => {
                      e.stopPropagation()
                      void select(m)
                    }}
                    title="Plan the fit for this model and choose how to load it"
                    data-testid="plan-model"
                  >
                    Load…
                  </button>
                  <button
                    className="row-action danger"
                    onClick={(e) => {
                      e.stopPropagation()
                      setPendingDelete(m)
                    }}
                    title={`Delete ${m.filename} from disk`}
                    aria-label={`Delete ${m.filename}`}
                    data-testid="delete-model"
                  >
                    <Icon name="trash" size={15} />
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/*
        * The plan opens over the library rather than unrolling beneath it.
        *
        * It used to append a heading, a notes card, a plan card and sometimes a two-up grid of
        * alternatives to the bottom of the page. On a machine with one model that is most of a
        * screenful appearing below the fold, and the button that triggered it said "Planning
        * below…" because there was no other way to tell you where the result had gone. Choosing
        * how to load a model is a decision with a beginning and an end, so it gets a surface
        * with a beginning and an end.
        */}
      {selected && (
        <div className="overlay" onClick={() => !busy && setSelected(null)} data-testid="plan-overlay">
          <div
            className="modal plan-modal"
            role="dialog"
            aria-modal="true"
            aria-label={`Load ${selected.filename}`}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="plan-head">
              <div className="plan-head-text">
                <h2 className="truncate" title={selected.filename}>
                  {displayModelName(selected.filename)}
                </h2>
                <div className="plan-head-sub">Fitted to the memory your graphics cards have free right now</div>
              </div>
              <button
                className="row-action"
                onClick={() => setSelected(null)}
                disabled={busy}
                title="Close"
                aria-label="Close"
                data-testid="plan-close"
              >
                <Icon name="close" size={15} />
              </button>
            </div>

            <div className="plan-body">
              {error && (
                <div className="card error-card">
                  <span className="badge bad">error</span> {error}
                </div>
              )}

              {/* The fit is recomputed on open, so the first paint has nothing to show yet. */}
              {!fit && (
                <div className="plan-pending">
                  <Spinner size={16} />
                  <span>Measuring how this fits on your cards…</span>
                </div>
              )}

              {fit && 'error' in fit && (
                <div className="card">
                  <span className="badge bad">cannot plan</span> {fit.error}
                </div>
              )}

              {fit && !('error' in fit) && (
                <>
                  {fit.needsUserChoice && (
                    <div className="card note">
                      Your context target can't be met with every layer on the GPUs. Nothing was changed for you: pick the
                      tradeoff you prefer.
                    </div>
                  )}
                  {plans.length > 1 && <PlanPicker plans={plans} index={planIndex} onPick={setPlanIndex} />}
                  {current && <PlanDetail plan={current} />}

                  {fit.notes.length > 0 && (
                    <details className="plan-more fit-notes">
                      <summary>
                        How this was planned
                        <Icon name="chevron" size={13} />
                      </summary>
                      <ul>
                        {fit.notes.map((n, i) => (
                          <li key={i}>{n}</li>
                        ))}
                      </ul>
                    </details>
                  )}

                  <AdvancedPlacement model={selected} value={overrides} onApply={(o) => void replan(o)} busy={busy} />
                </>
              )}
            </div>

            {/* One decision at the foot: the plan picked above. */}
            {current && (
              <div className="plan-foot">
                <button onClick={() => setSelected(null)} disabled={busy}>
                  Cancel
                </button>
                <button className="primary" disabled={busy} onClick={() => void load(current)}>
                  {busy ? <Spinner size={13} /> : null}
                  {busy ? 'Loading…' : 'Load with this plan'}
                </button>
              </div>
            )}
          </div>
        </div>
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
              {/* Named, because a model can now live in a folder of the user's own rather than the app's. */}
              <strong>{pendingDelete?.filename}</strong> will be erased from{' '}
              <code>{pendingDelete?.path.replace(/[\\/][^\\/]*$/, '')}</code>, freeing{' '}
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
