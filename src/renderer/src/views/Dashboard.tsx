import type { HardwareSnapshot, ModelRecord } from '@shared/types'

function fmt(n: number): string {
  if (n < 0) return 'unknown'
  const gb = n / 1024 ** 3
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(n / 1024 ** 2).toFixed(0)} MB`
}

export default function Dashboard({
  hardware,
  models,
  loaded
}: {
  hardware: HardwareSnapshot | null
  models: ModelRecord[]
  loaded: { model: string; port: number } | null
}): JSX.Element {
  return (
    <>
      <h1>Dashboard</h1>
      <p className="subtitle">
        Detected at runtime — nothing about your hardware is assumed or hardcoded.
      </p>

      {!hardware && <div className="card">Detecting hardware…</div>}

      {hardware && (
        <>
          <div className="grid-2">
            {hardware.gpus.length === 0 && (
              <div className="card">
                <div className="card-title">No GPU detected</div>
                <div className="dim">Inference will run on CPU.</div>
              </div>
            )}

            {hardware.gpus.map((g) => {
              const used = g.freeVram >= 0 ? g.totalVram - g.freeVram : 0
              const pct = g.totalVram > 0 ? Math.min(100, (used / g.totalVram) * 100) : 0
              return (
                <div className="card" key={`${g.index}-${g.name}`}>
                  <div className="card-title">
                    {g.name}
                    <span className={`badge ${g.freeIsMeasured ? 'good' : 'warn'}`}>
                      {g.freeIsMeasured ? 'measured' : 'estimated'}
                    </span>
                  </div>
                  <div className="meter">
                    <span style={{ width: `${pct}%` }} />
                  </div>
                  <dl className="kv" style={{ marginTop: 10 }}>
                    <dt>VRAM total</dt>
                    <dd>{fmt(g.totalVram)}</dd>
                    <dt>VRAM free</dt>
                    <dd>{fmt(g.freeVram)}</dd>
                    <dt>Utilisation</dt>
                    <dd>{g.utilisation >= 0 ? `${g.utilisation}%` : 'unknown'}</dd>
                  </dl>
                  {!g.freeIsMeasured && (
                    <p className="faint" style={{ marginBottom: 0, marginTop: 8 }}>
                      Free VRAM cannot be measured on this adapter; the auto-fit engine budgets
                      conservatively against 85% of total.
                    </p>
                  )}
                </div>
              )
            })}

            <div className="card">
              <div className="card-title">System</div>
              <dl className="kv">
                <dt>Backend</dt>
                <dd>{hardware.backend.toUpperCase()}</dd>
                <dt>CPU</dt>
                <dd>{hardware.cpuName}</dd>
                <dt>Threads</dt>
                <dd>{hardware.cpuThreads}</dd>
                <dt>RAM total</dt>
                <dd>{fmt(hardware.totalRam)}</dd>
                <dt>RAM free</dt>
                <dd>{fmt(hardware.freeRam)}</dd>
              </dl>
            </div>

            <div className="card">
              <div className="card-title">Library</div>
              <dl className="kv">
                <dt>Models</dt>
                <dd>{models.length}</dd>
                <dt>Total size</dt>
                <dd>{fmt(models.reduce((a, m) => a + m.bytes, 0))}</dd>
                <dt>Loaded</dt>
                <dd>{loaded ? loaded.model : 'none'}</dd>
              </dl>
            </div>
          </div>
        </>
      )}
    </>
  )
}
