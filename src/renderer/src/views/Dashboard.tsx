import { useEffect, useState } from 'react'
import type { HardwareSnapshot, ModelRecord } from '@shared/types'
import { fmtBytes, invoke } from '../lib/api'
import type { LoadedModel, View } from '../App'

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
  const [missing, setMissing] = useState<string[]>([])

  useEffect(() => {
    const tick = async (): Promise<void> => {
      try {
        setLive(await invoke<LiveStats>('stats:live'))
      } catch {
        /* transient */
      }
    }
    void tick()
    void invoke<History>('stats:history').then(setHistory).catch(() => undefined)
    void invoke<string[]>('runtime:missing-binaries').then(setMissing).catch(() => undefined)
    const timer = setInterval(tick, 2000)
    return () => clearInterval(timer)
  }, [])

  return (
    <>
      <h1>Dashboard</h1>
      <p className="subtitle">Everything here is measured at runtime. No hardware values are assumed.</p>

      {missing.length > 0 && (
        <div className="card" style={{ borderColor: '#5a4515' }}>
          <div className="card-title">
            Setup incomplete <span className="badge warn">{missing.length} missing</span>
          </div>
          <div className="dim">
            These bundled components have not been fetched, so the features that need them are unavailable:
            <div className="mono" style={{ margin: '8px 0' }}>{missing.join(', ')}</div>
            Run <code className="mono">npm run fetch-vendor</code> to download them.
          </div>
        </div>
      )}

      <div className="grid-2">
        {(hardware?.gpus ?? []).map((g) => {
          const used = g.freeVram >= 0 ? g.totalVram - g.freeVram : 0
          const pct = g.totalVram > 0 ? Math.min(100, (used / g.totalVram) * 100) : 0
          return (
            <div className="card" key={`${g.index}-${g.name}`}>
              <div className="card-title">
                <span className="truncate">{g.name}</span>
                <span className={`badge ${g.freeIsMeasured ? 'good' : 'warn'}`}>
                  {g.freeIsMeasured ? 'measured' : 'estimated'}
                </span>
              </div>
              <div className="meter">
                <span style={{ width: `${pct}%` }} />
              </div>
              <div className="row" style={{ justifyContent: 'space-between', marginTop: 6, fontSize: 11 }}>
                <span className="faint">{fmtBytes(used)} used</span>
                <span className="faint">{fmtBytes(g.freeVram)} free of {fmtBytes(g.totalVram)}</span>
              </div>
              {g.utilisation >= 0 && (
                <div className="faint" style={{ marginTop: 4, fontSize: 11 }}>GPU {g.utilisation}% busy</div>
              )}
            </div>
          )
        })}

        {hardware && hardware.gpus.length === 0 && (
          <div className="card">
            <div className="card-title">No GPU detected</div>
            <div className="dim">Inference will run on CPU. Expect a few tokens per second on larger models.</div>
          </div>
        )}

        <div className="card">
          <div className="card-title">System</div>
          <dl className="kv">
            <dt>Backend</dt>
            <dd>{hardware?.backend.toUpperCase() ?? '—'}</dd>
            <dt>CPU</dt>
            <dd className="truncate">{hardware?.cpuName ?? '—'}</dd>
            <dt>Threads</dt>
            <dd>{hardware?.cpuThreads ?? '—'}</dd>
            <dt>RAM</dt>
            <dd>{fmtBytes(hardware?.freeRam)} free of {fmtBytes(hardware?.totalRam)}</dd>
          </dl>
        </div>

        <div className="card">
          <div className="card-title">
            Inference
            {loaded && <span className="badge good">live</span>}
          </div>
          {loaded ? (
            <>
              <dl className="kv">
                <dt>Model</dt>
                <dd className="truncate">{loaded.model}</dd>
                <dt>Speed</dt>
                <dd>{live?.tokensPerSecond ? `${live.tokensPerSecond.toFixed(1)} tok/s` : '—'}</dd>
                <dt>First token</dt>
                <dd>{live?.ttftMs != null ? `${live.ttftMs} ms` : '—'}</dd>
                <dt>Context</dt>
                <dd>
                  {(live?.contextUsed ?? 0).toLocaleString()} / {loaded.plan.contextLength.toLocaleString()}
                </dd>
              </dl>
              <div className="meter" style={{ marginTop: 8 }}>
                <span style={{ width: `${live?.kvFillPercent ?? 0}%` }} />
              </div>
              <div className="faint" style={{ fontSize: 11, marginTop: 4 }}>
                KV cache fill — reserved in full at load, so it cannot overflow mid-chat
              </div>
            </>
          ) : (
            <>
              <div className="dim">No model loaded.</div>
              <button className="primary" style={{ marginTop: 10 }} onClick={() => onNavigate('library')}>
                Choose a model
              </button>
            </>
          )}
        </div>

        <div className="card">
          <div className="card-title">Library</div>
          <dl className="kv">
            <dt>Models</dt>
            <dd>{models.length}</dd>
            <dt>On disk</dt>
            <dd>{fmtBytes(models.reduce((a, m) => a + m.bytes, 0))}</dd>
            <dt>Multimodal</dt>
            <dd>{models.filter((m) => m.caps.vision || m.caps.audio).length}</dd>
          </dl>
          {models.length === 0 && (
            <button className="primary" style={{ marginTop: 10 }} onClick={() => onNavigate('discover')}>
              Find a model
            </button>
          )}
        </div>

        {history && history.totalTokensOut > 0 && (
          <div className="card">
            <div className="card-title">All time</div>
            <dl className="kv">
              <dt>Tokens generated</dt>
              <dd>{history.totalTokensOut.toLocaleString()}</dd>
              <dt>API requests</dt>
              <dd>{history.totalRequests.toLocaleString()}</dd>
              <dt>Active time</dt>
              <dd>{history.activeHours.toFixed(1)} h</dd>
            </dl>
          </div>
        )}
      </div>
    </>
  )
}
