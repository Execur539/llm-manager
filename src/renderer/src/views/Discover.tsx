import { useEffect, useState } from 'react'
import { fmtBytes, invoke, on } from '../lib/api'

interface HfModelSummary {
  id: string
  downloads: number
  likes: number
  updatedAt: string
  gated: boolean
}

interface HfFile {
  filename: string
  bytes: number
  quant: string | null
  isMmproj: boolean
  shard: { index: number; total: number } | null
}

interface Recommendation {
  filename: string
  reason: string
  predictedContext: number
  fitsFullyOnGpu: boolean
}

interface DownloadItem {
  id: string
  filename: string
  bytesTotal: number
  bytesDone: number
  status: string
  error: string | null
  speed: number
}

export default function Discover({ onDownloaded }: { onDownloaded: () => Promise<void> }): JSX.Element {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<HfModelSummary[]>([])
  const [searching, setSearching] = useState(false)
  const [selected, setSelected] = useState<string | null>(null)
  const [files, setFiles] = useState<HfFile[]>([])
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null)
  const [showAll, setShowAll] = useState(false)
  const [downloads, setDownloads] = useState<DownloadItem[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    void invoke<DownloadItem[]>('downloads:list').then(setDownloads).catch(() => undefined)
    const off = on<DownloadItem[]>('downloads:update', (list) => {
      setDownloads(list)
      if (list.some((d) => d.status === 'done')) void onDownloaded()
    })
    return off
  }, [onDownloaded])

  const search = async (): Promise<void> => {
    if (!query.trim()) return
    setSearching(true)
    setError(null)
    setSelected(null)
    try {
      setResults(await invoke<HfModelSummary[]>('hf:search', query))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSearching(false)
    }
  }

  const openRepo = async (repo: string): Promise<void> => {
    setSelected(repo)
    setFiles([])
    setRecommendation(null)
    setError(null)
    setShowAll(false)
    try {
      const result = await invoke<{ files: HfFile[]; recommendation: Recommendation | null }>('hf:files', repo)
      setFiles(result.files)
      setRecommendation(result.recommendation)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const download = async (filename: string): Promise<void> => {
    if (!selected) return
    setError(null)
    try {
      await invoke('hf:download', selected, filename)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const active = downloads.filter((d) => ['queued', 'downloading', 'paused'].includes(d.status))
  const shownFiles = showAll ? files : files.filter((f) => !f.isMmproj && (!f.shard || f.shard.index === 1))

  return (
    <>
      <h1>Find a model</h1>
      <p className="subtitle">
        Searches HuggingFace for GGUF repositories. Nothing is hardcoded, so models released today show up today.
      </p>

      <div className="row" style={{ marginBottom: 16 }}>
        <input
          type="text"
          placeholder="e.g. qwen3.8 27b, llama, gemma…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && void search()}
          style={{ flex: 1 }}
        />
        <button className="primary" onClick={() => void search()} disabled={searching}>
          {searching ? 'Searching…' : 'Search'}
        </button>
      </div>

      {error && (
        <div className="card" style={{ borderColor: '#5c2626' }}>
          <span className="badge bad">error</span> {error}
        </div>
      )}

      {active.length > 0 && (
        <div className="card">
          <div className="card-title">Downloads</div>
          {active.map((d) => {
            const pct = d.bytesTotal > 0 ? (d.bytesDone / d.bytesTotal) * 100 : 0
            return (
              <div key={d.id} style={{ marginBottom: 10 }}>
                <div className="row" style={{ justifyContent: 'space-between', fontSize: 12 }}>
                  <span className="truncate">{d.filename}</span>
                  <span className="faint">
                    {fmtBytes(d.bytesDone)} / {fmtBytes(d.bytesTotal)}
                    {d.speed > 0 && ` · ${fmtBytes(d.speed)}/s`}
                  </span>
                </div>
                <div className="meter" style={{ marginTop: 4 }}>
                  <span style={{ width: `${pct}%` }} />
                </div>
                <div className="row" style={{ marginTop: 6 }}>
                  <span className="badge">{d.status}</span>
                  {d.status === 'downloading' && <button onClick={() => void invoke('downloads:pause', d.id)}>Pause</button>}
                  {d.status === 'paused' && <button onClick={() => void invoke('downloads:resume', d.id)}>Resume</button>}
                  <button className="danger" onClick={() => void invoke('downloads:cancel', d.id)}>Cancel</button>
                </div>
                {d.error && <div className="badge bad" style={{ marginTop: 4 }}>{d.error}</div>}
              </div>
            )
          })}
          <div className="faint" style={{ fontSize: 11 }}>
            Downloads resume where they stopped, even after the app is closed.
          </div>
        </div>
      )}

      {!selected && (
        <div className="list">
          {results.map((r) => (
            <div className="card row-card" key={r.id} onClick={() => void openRepo(r.id)}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="truncate" style={{ fontWeight: 600 }}>{r.id}</div>
                <div className="faint" style={{ fontSize: 11 }}>
                  {r.downloads.toLocaleString()} downloads · {r.likes} likes
                  {r.updatedAt && ` · updated ${new Date(r.updatedAt).toLocaleDateString()}`}
                </div>
              </div>
              {r.gated && <span className="badge warn">gated</span>}
            </div>
          ))}
          {!results.length && !searching && (
            <div className="empty">Search for a model to get started.</div>
          )}
        </div>
      )}

      {selected && (
        <>
          <div className="row" style={{ marginBottom: 12 }}>
            <button onClick={() => setSelected(null)}>← Back to results</button>
            <h2 style={{ margin: 0, fontSize: 15 }}>{selected}</h2>
          </div>

          {recommendation && (
            <div className="card" style={{ borderColor: recommendation.fitsFullyOnGpu ? '#1f4a33' : '#5a4515' }}>
              <div className="card-title">
                Recommended for your hardware
                <span className={`badge ${recommendation.fitsFullyOnGpu ? 'good' : 'warn'}`}>
                  {recommendation.fitsFullyOnGpu ? 'fits in VRAM' : 'partial offload'}
                </span>
              </div>
              <div className="mono" style={{ marginBottom: 6 }}>{recommendation.filename}</div>
              <div className="dim">{recommendation.reason}</div>
              <button className="primary" style={{ marginTop: 10 }} onClick={() => void download(recommendation.filename)}>
                Download this one
              </button>
            </div>
          )}

          <div className="row" style={{ margin: '14px 0 8px' }}>
            <h3 style={{ margin: 0, fontSize: 13 }}>All variants ({files.length})</h3>
            <label className="faint row" style={{ gap: 6, fontSize: 12 }}>
              <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />
              include projectors and shards
            </label>
          </div>

          <div className="list">
            {shownFiles.map((f) => (
              <div className="card row-card" key={f.filename}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="truncate mono" style={{ fontSize: 12 }}>{f.filename}</div>
                  <div className="faint" style={{ fontSize: 11 }}>
                    {fmtBytes(f.bytes)}
                    {f.quant && ` · ${f.quant}`}
                    {f.shard && ` · part ${f.shard.index} of ${f.shard.total}`}
                    {f.isMmproj && ' · vision projector'}
                  </div>
                </div>
                <button onClick={() => void download(f.filename)}>Download</button>
              </div>
            ))}
            {!files.length && <div className="empty">Loading files…</div>}
          </div>

          <p className="faint" style={{ fontSize: 11, marginTop: 10 }}>
            Multimodal models need their mmproj companion — it is fetched automatically alongside the model.
          </p>
        </>
      )}
    </>
  )
}
