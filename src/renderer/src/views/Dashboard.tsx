import { useEffect, useState } from 'react'
import type { HardwareSnapshot, ModelRecord } from '@shared/types'
import { fmtBytes, invoke, displayModelName } from '../lib/api'
import { Spinner } from '../components/Spinner'
import type { LoadedModel, View } from '../App'
import { Toolbar } from '../lib/shell'

interface LiveStats {
  tokensPerSecond: number
  ttftMs: number | null
  contextUsed: number
  contextLength: number
  kvFillPercent: number
}

interface History {
  totalTokensOut: number
  totalRequests: number
  activeHours: number
  byModel: { modelId: string; tokensOut: number; avgTokensPerSecond: number }[]
}

export default function Dashboard({
  hardware,
  models,
  loaded,
  onNavigate
}: {
  hardware: HardwareSnapshot | null
  models: ModelRecord[]
  loaded: LoadedModel | null
  onNavigate: (v: View) => void
}): JSX.Element {
  const [live, setLive] = useState<LiveStats | null>(null)
  const [history, setHistory] = useState<History | null>(null)
  const [vendor, setVendor] = useState<{ root: string; rootExists: boolean; missing: string[]; present: string[] } | null>(null)

  /*
   * Two cadences, because the numbers here move at two different speeds.
   *
   * Everything except the live meters used to be fetched once, on mount, and then left — so
   * totals, per-model averages and the setup diagnostics were accurate as of whenever the view
   * happened to be opened, with nothing to suggest otherwise. Opening the dashboard and watching
   * it was indistinguishable from a machine doing nothing at all.
   *
   * The live meters keep their faster tick: tokens per second and time-to-first-token describe
   * a response in progress, and at eight seconds a short answer would begin and end between two
   * samples. The hardware figures behind them are refreshed by the main process on the same
   * eight-second beat, so they arrive here without this view having to ask.
   */
  useEffect(() => {
    const live = async (): Promise<void> => {
      try {
        setLive(await invoke<LiveStats>('stats:live'))
      } catch {
        /* transient */
      }
    }
    const totals = async (): Promise<void> => {
      await Promise.all([
        invoke<History>('stats:history').then(setHistory).catch(() => undefined),
        invoke<typeof vendor>('runtime:vendor-diagnostics').then(setVendor).catch(() => undefined)
      ])
    }
    void live()
    void totals()
    const fast = setInterval(live, 2000)
    const slow = setInterval(totals, 8000)
    return () => {
      clearInterval(fast)
      clearInterval(slow)
    }
  }, [])

  return (
    <>
      <Toolbar title="Dashboard" />

      {vendor && vendor.missing.length > 0 && (
        <div className="card" style={{ borderColor: '#5a4515' }}>
          <div className="card-title">
            Setup incomplete <span className="badge warn">{vendor.missing.length} missing</span>
          </div>
          <div className="dim">
            These bundled components could not be found, so the features that need them are unavailable:
            <div className="mono" style={{ margin: '8px 0' }}>{vendor.missing.join(', ')}</div>
            <div style={{ marginTop: 8 }}>
              Searched: <span className="mono">{vendor.root}</span>{' '}
              <span className={`badge ${vendor.rootExists ? 'good' : 'bad'}`}>
                {vendor.rootExists ? 'exists' : 'not found'}
              </span>
            </div>
            {vendor.rootExists && vendor.present.length > 0 && (
              <div className="faint" style={{ marginTop: 4, fontSize: 11 }}>
                Found there: {vendor.present.join(', ')}
              </div>
            )}
            <div style={{ marginTop: 8 }}>
              {vendor.rootExists
                ? 'Some components are absent from that directory.'
                : 'That directory does not exist.'}{' '}
              Run <code className="mono">npm run fetch-vendor</code> to download them.
            </div>
          </div>
        </div>
      )}

      {/*
        * The model, first and large: whether one is running, and how it is doing. Everything else on
        * this page is the machine it runs on.
        */}
      {loaded ? (
        <section className="card dash-hero" aria-label="Inference">
          <span className="sr-only">Inference</span>
          <div className="dash-hero-head">
            <span className="dash-status">
              <span className="pulse-dot" aria-hidden="true" />
              Running
            </span>
            <h2 className="dash-model truncate" title={loaded.model}>
              {displayModelName(loaded.model)}
            </h2>
          </div>
          <div className="dash-stats">
            <div className="stat">
              <div className="stat-label">Speed</div>
              <div className="stat-value">
                {live?.tokensPerSecond ? live.tokensPerSecond.toFixed(1) : '—'}
                <span className="stat-unit">tok/s</span>
              </div>
            </div>
            <div className="stat">
              <div className="stat-label">First token</div>
              <div className="stat-value">
                {live?.ttftMs != null ? live.ttftMs.toLocaleString() : '—'}
                <span className="stat-unit">ms</span>
              </div>
            </div>
            <div className="stat">
              <div className="stat-label">Context in use</div>
              <div className="stat-value">
                {Math.round(live?.kvFillPercent ?? 0)}
                <span className="stat-unit">%</span>
              </div>
              <div className="stat-sub">
                {(live?.contextUsed ?? 0).toLocaleString()} of {loaded.plan.contextLength.toLocaleString()} tokens
              </div>
            </div>
            <div className="stat">
              <div className="stat-label">Layers on GPU</div>
              <div className="stat-value">
                {loaded.plan.gpuLayers}
                <span className="stat-unit">of {loaded.plan.totalLayers}</span>
              </div>
            </div>
          </div>
          <div className="dash-actions">
            <button className="primary" onClick={() => onNavigate('chat')}>
              Open chat
            </button>
            <button onClick={() => onNavigate('agent')}>Open agent</button>
          </div>
        </section>
      ) : (
        <section className="card dash-hero idle" aria-label="Inference">
          <span className="sr-only">Inference</span>
          <div className="dash-hero-head">
            <span className="dash-status idle">Idle</span>
            <h2 className="dash-model">No model loaded.</h2>
          </div>
          <p className="dash-lede">
            {models.length
              ? `Load one of your ${models.length} ${models.length === 1 ? 'model' : 'models'} to chat, run the agent, or serve the API.`
              : 'Find a model to download, then load it here to chat, run the agent, or serve the API.'}
          </p>
          <div className="dash-actions">
            {models.length > 0 && (
              <button className="primary" onClick={() => onNavigate('library')}>
                Choose a model
              </button>
            )}
            <button className={models.length ? '' : 'primary'} onClick={() => onNavigate('discover')}>
              Find a model
            </button>
          </div>
        </section>
      )}

      {/* The cards the models run on, as a list of meters rather than a wall of boxes. */}
      <section className="dash-section">
        <h2 className="section-title">Graphics</h2>
        {hardware?.detection?.state === 'provisional' && (
          <div className="card note" data-testid="gpu-detecting">
            <Spinner size={13} />
            <span>
              {hardware.gpus.length ? 'Confirming GPUs. ' : 'Detecting GPUs. '}
              {hardware.detection.reason
                ? `The GPU driver has not answered yet (${hardware.detection.reason}). `
                : 'The GPU driver has not answered yet. '}
              Retrying automatically; plans made before it answers may leave GPUs out.
            </span>
          </div>
        )}
        {hardware && hardware.gpus.length === 0 && hardware.detection?.state !== 'provisional' && (
          <div className="card dash-empty">
            <strong>No GPU detected.</strong> Models will run on the processor, at a few tokens per second for the larger ones.
          </div>
        )}
        {(hardware?.gpus.length ?? 0) > 0 && (
          <div className="card gpu-list">
            {(hardware?.gpus ?? []).map((g) => {
              const measured = g.freeVram >= 0
              const used = measured ? g.totalVram - g.freeVram : 0
              const pct = measured && g.totalVram > 0 ? Math.min(100, (used / g.totalVram) * 100) : 0
              const unusable = hardware?.backend === 'cuda' && g.vendor !== 'nvidia'
              return (
                <div className={`gpu-row${unusable ? ' unusable' : ''}`} key={`${g.index}-${g.name}`}>
                  <div className="gpu-row-head">
                    <span className="gpu-name truncate" title={g.name}>
                      {g.name.replace(/^NVIDIA GeForce /, '')}
                    </span>
                    <span className="gpu-figures">
                      {measured
                        ? `${fmtBytes(used)} of ${fmtBytes(g.totalVram)}`
                        : `${fmtBytes(g.totalVram)}, use not reported`}
                      {g.utilisation >= 0 && <span className="gpu-busy">{g.utilisation}% busy</span>}
                    </span>
                  </div>
                  <div className={`meter${measured ? '' : ' unknown'}`} title={g.freeIsMeasured ? 'Read from the driver' : 'Estimated from the total; this driver does not report free memory'}>
                    {measured && <span style={{ width: `${pct}%` }} />}
                  </div>
                  {unusable && <div className="gpu-note">Shares system memory; not used for models with the CUDA backend.</div>}
                </div>
              )
            })}
          </div>
        )}
      </section>

      <section className="dash-section">
        <h2 className="section-title">System</h2>
        <div className="dash-tiles">
          <div className="card dash-tile">
            <div className="stat-label">Memory</div>
            <div className="stat-value small">
              {fmtBytes(Math.max(0, (hardware?.totalRam ?? 0) - (hardware?.freeRam ?? 0)))}
              <span className="stat-unit">of {fmtBytes(hardware?.totalRam)} in use</span>
            </div>
            <div className="meter">
              <span
                style={{
                  width: `${hardware?.totalRam ? Math.min(100, ((hardware.totalRam - (hardware.freeRam ?? 0)) / hardware.totalRam) * 100) : 0}%`
                }}
              />
            </div>
          </div>
          <div className="card dash-tile">
            <div className="stat-label">Processor</div>
            <div className="dash-tile-text" title={hardware?.cpuName ?? undefined}>
              {hardware?.cpuName ?? '—'}
            </div>
            <div className="stat-sub">
              {hardware?.cpuThreads ?? '—'} threads · {hardware?.backend.toUpperCase() ?? '—'} backend
            </div>
          </div>
          <div className="card dash-tile">
            <div className="stat-label">Library</div>
            <div className="stat-value small">
              {models.length}
              <span className="stat-unit">{models.length === 1 ? 'model' : 'models'}</span>
            </div>
            <div className="stat-sub">
              {fmtBytes(models.reduce((a, m) => a + m.bytes, 0))} on disk ·{' '}
              {models.filter((m) => m.caps.vision || m.caps.audio).length} multimodal
            </div>
          </div>
          {history && history.totalTokensOut > 0 && (
            <div className="card dash-tile">
              <div className="stat-label">All time</div>
              <div className="stat-value small">
                {history.totalTokensOut >= 1e6
                  ? `${(history.totalTokensOut / 1e6).toFixed(1)}M`
                  : history.totalTokensOut.toLocaleString()}
                <span className="stat-unit">tokens</span>
              </div>
              <div className="stat-sub">
                {history.totalRequests.toLocaleString()} API requests · {history.activeHours.toFixed(1)} h active
              </div>
            </div>
          )}
        </div>
      </section>
    </>
  )
}
