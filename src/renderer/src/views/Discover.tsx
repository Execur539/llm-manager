import { useEffect, useRef, useState } from 'react'
import { fmtBytes, invoke, on, fmtRelative } from '../lib/api'
import Icon from '../components/Icon'
import { Skeleton, Spinner } from '../components/Spinner'

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

/**
 * Compact counts for HuggingFace figures.
 *
 * "6,674,515 downloads" is precise and unreadable; at a glance the only question is the order of
 * magnitude. The exact number stays available as a tooltip.
 */
function fmtCount(n: number): string {
  if (!Number.isFinite(n) || n < 0) return '—'
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n >= 10_000_000 ? 0 : 1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(n >= 10_000 ? 0 : 1)}k`
  return String(n)
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

  /** Downloads already seen finished, so a completion is reacted to once rather than forever. */
  const settled = useRef<Set<string>>(new Set())

  useEffect(() => {
    void invoke<DownloadItem[]>('downloads:list').then((list) => {
      setDownloads(list)
      // Anything already done when the view mounts is history, not news.
      for (const d of list) if (d.status === 'done') settled.current.add(d.id)
    })

    /*
     * Rescan on a download *becoming* done, not on one *being* done.
     *
     * The queue emits `update` about twice a second while any transfer is running, and finished
     * rows stay in the table indefinitely — so `list.some(d => d.status === 'done')` is true
     * forever once a single model has ever been downloaded. Every tick therefore kicked off a
     * full library scan: a directory walk and a stat of every GGUF, twice a second, for the whole
     * of a twenty-gigabyte download.
     */
    const off = on<DownloadItem[]>('downloads:update', (list) => {
      setDownloads(list)
      const newlyDone = list.filter((d) => d.status === 'done' && !settled.current.has(d.id))
      for (const d of list) {
        if (d.status === 'done') settled.current.add(d.id)
        else settled.current.delete(d.id)
      }
      if (newlyDone.length) void onDownloaded()
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

  // 'verifying' belongs here too: the bytes have arrived but the download is not finished, and
  // dropping the row off the list mid-hash looks like it silently vanished.
  const active = downloads.filter((d) => ['queued', 'downloading', 'verifying', 'paused'].includes(d.status))
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
        <button className="primary" onClick={() => void search()} disabled={searching} data-testid="search-models">
          {searching ? <Spinner size={13} /> : <Icon name="search" size={14} />}
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
          {results.map((r) => {
            const [owner, ...rest] = r.id.split('/')
            const name = rest.join('/') || r.id
            return (
              <button
                type="button"
                className="card row-card repo-card"
                key={r.id}
                onClick={() => void openRepo(r.id)}
                title={`Open ${r.id}`}
              >
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="truncate repo-name">
                    {rest.length > 0 && <span className="repo-owner">{owner}/</span>}
                    {name}
                  </div>
                  <div className="repo-meta">
                    <span title={`${r.downloads.toLocaleString()} downloads`}>
                      <Icon name="download" size={11} /> {fmtCount(r.downloads)}
                    </span>
                    <span title={`${r.likes.toLocaleString()} likes`}>
                      <Icon name="star" size={11} /> {fmtCount(r.likes)}
                    </span>
                    {r.updatedAt && (
                      <span title={new Date(r.updatedAt).toLocaleString()}>
                        updated {fmtRelative(new Date(r.updatedAt).getTime())}
                      </span>
                    )}
                  </div>
                </div>
                {r.gated && <span className="badge warn">gated</span>}
                {/* The whole row is clickable; without a mark that is not obvious. */}
                <Icon name="search" size={14} className="repo-go" />
              </button>
            )
          })}
          {searching && !results.length && <Skeleton rows={4} height={58} />}
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
