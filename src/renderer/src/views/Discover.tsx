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

interface HfVariant {
  id: string
  label: string
  quant: string | null
  bytes: number
  parts: HfFile[]
  complete: boolean
  missing: number[]
  kind: 'model' | 'mmproj' | 'mtp'
}

/** What a variant's header says about it, read before download. */
interface VariantInfo {
  mtpLayers?: number
  expertCount?: number
  expertUsedCount?: number
}

interface Recommendation {
  variantId: string
  label: string
  reason: string
  predictedContext: number
  fitsFullyOnGpu: boolean
}

interface DownloadItem {
  id: string
  repo: string | null
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

/** The model a downloaded file belongs to: its name with any split-part suffix removed. */
function downloadGroupKey(d: DownloadItem): string {
  return `${d.repo ?? ''}::${d.filename.replace(/-\d{5}-of-\d{5}(?=\.gguf$)/i, '')}`
}

interface DownloadGroup {
  key: string
  label: string
  items: DownloadItem[]
  parts: number
  partsDone: number
  bytesDone: number
  bytesTotal: number
  speed: number
  status: string
  error: string | null
}

/**
 * One row per model, however many files it arrives as.
 *
 * Every part is its own queue entry, so a split model showed as several unrelated progress bars
 * that each had to be paused or cancelled separately. Grouped here, the row sums them and its
 * buttons act on all of them. Only the newest entry per file counts, so an earlier cancelled
 * attempt does not double the totals.
 */
function groupDownloads(list: DownloadItem[]): DownloadGroup[] {
  const latest = new Map<string, DownloadItem>()
  for (const d of list) {
    const k = `${d.repo}::${d.filename}`
    if (!latest.has(k)) latest.set(k, d)
  }
  const groups = new Map<string, DownloadItem[]>()
  for (const d of latest.values()) {
    const key = downloadGroupKey(d)
    groups.set(key, [...(groups.get(key) ?? []), d])
  }
  // A failed part keeps its model on screen: the row is the only place to resume it from.
  const live = ['queued', 'downloading', 'verifying', 'paused', 'failed']
  const out: DownloadGroup[] = []
  for (const [key, items] of groups) {
    if (!items.some((d) => live.includes(d.status))) continue
    const has = (st: string): boolean => items.some((d) => d.status === st)
    const status = has('downloading')
      ? 'downloading'
      : has('verifying')
        ? 'verifying'
        : has('queued')
          ? 'queued'
          : has('paused')
            ? 'paused'
            : 'failed'
    out.push({
      key,
      label: key.split('::')[1] ?? key,
      items,
      parts: items.length,
      partsDone: items.filter((d) => d.status === 'done').length,
      bytesDone: items.reduce((a, d) => a + (d.status === 'done' ? d.bytesTotal : d.bytesDone), 0),
      bytesTotal: items.reduce((a, d) => a + d.bytesTotal, 0),
      speed: items.reduce((a, d) => a + (d.speed || 0), 0),
      status,
      error: items.find((d) => d.error)?.error ?? null
    })
  }
  return out
}

export default function Discover({ onDownloaded }: { onDownloaded: () => Promise<void> }): JSX.Element {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<HfModelSummary[]>([])
  const [searching, setSearching] = useState(false)
  const [selected, setSelected] = useState<string | null>(null)
  const [files, setFiles] = useState<HfFile[]>([])
  const [variants, setVariants] = useState<HfVariant[]>([])
  const [info, setInfo] = useState<Record<string, VariantInfo | null>>({})
  /** The repo whose headers are being read; a peek for any other one is dropped. */
  const current = useRef<string | null>(null)
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
    setVariants([])
    setInfo({})
    setRecommendation(null)
    setError(null)
    setShowAll(false)
    current.current = repo
    try {
      const result = await invoke<{ files: HfFile[]; variants: HfVariant[]; recommendation: Recommendation | null }>(
        'hf:files',
        repo
      )
      setFiles(result.files)
      setVariants(result.variants)
      setRecommendation(result.recommendation)
      void peekAll(repo, result.variants.filter((v) => v.kind === 'model' && v.complete))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  /*
   * Read each variant's header, three at a time, for the MTP and MoE badges.
   *
   * A range request each, so they are not fired all at once at a repo with a dozen quants, and a
   * repo left mid-way stops being asked about.
   */
  const peekAll = async (repo: string, list: HfVariant[]): Promise<void> => {
    const queue = [...list]
    const worker = async (): Promise<void> => {
      for (let v = queue.shift(); v; v = queue.shift()) {
        if (current.current !== repo) return
        const id = v.id
        const got = await invoke<VariantInfo | null>('hf:variant-info', repo, id).catch(() => null)
        if (current.current !== repo) return
        setInfo((prev) => ({ ...prev, [id]: got }))
      }
    }
    await Promise.all([worker(), worker(), worker()])
  }

  const download = async (variantId: string): Promise<void> => {
    if (!selected) return
    setError(null)
    try {
      await invoke('hf:download', selected, variantId)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  // 'verifying' belongs here too: the bytes have arrived but the download is not finished, and
  // dropping the row off the list mid-hash looks like it silently vanished.
  const active = groupDownloads(downloads)
  const modelVariants = variants.filter((v) => v.kind === 'model')
  const companions = variants.filter((v) => v.kind !== 'model')
  const act = (g: DownloadGroup, channel: string, from: string[]): void => {
    for (const d of g.items) if (from.includes(d.status)) void invoke(channel, d.id)
  }

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
          {active.map((g) => {
            const pct = g.bytesTotal > 0 ? (g.bytesDone / g.bytesTotal) * 100 : 0
            return (
              <div key={g.key} style={{ marginBottom: 10 }}>
                <div className="row" style={{ justifyContent: 'space-between', fontSize: 12 }}>
                  <span className="truncate">{g.label}</span>
                  <span className="faint">
                    {fmtBytes(g.bytesDone)} / {fmtBytes(g.bytesTotal)}
                    {g.speed > 0 && ` · ${fmtBytes(g.speed)}/s`}
                  </span>
                </div>
                <div className="meter" style={{ marginTop: 4 }}>
                  <span style={{ width: `${pct}%` }} />
                </div>
                <div className="row" style={{ marginTop: 6 }}>
                  <span className="badge">{g.status}</span>
                  {g.parts > 1 && (
                    <span className="badge">
                      {g.partsDone} of {g.parts} parts
                    </span>
                  )}
                  {['downloading', 'queued', 'verifying'].includes(g.status) && (
                    <button onClick={() => act(g, 'downloads:pause', ['downloading', 'queued'])}>Pause</button>
                  )}
                  {g.items.some((d) => d.status === 'paused' || d.status === 'failed') && (
                    <button onClick={() => act(g, 'downloads:resume', ['paused', 'failed'])}>Resume</button>
                  )}
                  <button
                    className="danger"
                    onClick={() => act(g, 'downloads:cancel', ['queued', 'downloading', 'verifying', 'paused'])}
                  >
                    Cancel
                  </button>
                </div>
                {g.error && <div className="badge bad" style={{ marginTop: 4 }}>{g.error}</div>}
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
                  {recommendation.fitsFullyOnGpu ? 'fits in VRAM' : 'uses system RAM'}
                </span>
              </div>
              <div className="mono" style={{ marginBottom: 6 }}>{recommendation.label}</div>
              <div className="dim">{recommendation.reason}</div>
              <button className="primary" style={{ marginTop: 10 }} onClick={() => void download(recommendation.variantId)}>
                Download this one
              </button>
            </div>
          )}

          <div className="row" style={{ margin: '14px 0 8px' }}>
            <h3 style={{ margin: 0, fontSize: 13 }}>All variants ({modelVariants.length})</h3>
            {companions.length > 0 && (
              <label className="faint row" style={{ gap: 6, fontSize: 12 }}>
                <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />
                show projectors and MTP modules
              </label>
            )}
          </div>

          <div className="list">
            {modelVariants.map((v) => {
              const vi = info[v.id]
              return (
                <div className="card row-card" key={v.id} data-testid="variant-row">
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
                      <span className="truncate mono" style={{ fontSize: 12 }} title={v.id}>
                        {v.label}
                      </span>
                      {(vi?.mtpLayers ?? 0) > 0 && (
                        <span
                          className="badge good"
                          title="Multi-token prediction is built into this file: the model drafts its next few tokens and checks them in the same pass, for faster generation with identical output. Nothing extra to download."
                        >
                          MTP
                        </span>
                      )}
                      {(vi?.expertCount ?? 0) > 0 && (
                        <span
                          className="badge"
                          title="Mixture of experts: only a few experts are read per token, so the rest can live in system RAM when VRAM runs out."
                        >
                          MoE {vi?.expertUsedCount ? `${vi.expertUsedCount}/` : ''}
                          {vi?.expertCount}
                        </span>
                      )}
                      {v.parts.length > 1 && <span className="badge">{v.parts.length} parts</span>}
                      {!v.complete && <span className="badge bad">incomplete on HuggingFace</span>}
                    </div>
                    <div className="faint" style={{ fontSize: 11 }}>
                      {fmtBytes(v.bytes)}
                      {v.quant && v.quant !== v.label && ` · ${v.quant}`}
                      {!v.complete && ` · part${v.missing.length === 1 ? '' : 's'} ${v.missing.join(', ')} missing`}
                    </div>
                  </div>
                  <button
                    onClick={() => void download(v.id)}
                    disabled={!v.complete}
                    title={v.parts.length > 1 ? `Downloads all ${v.parts.length} parts` : undefined}
                  >
                    Download
                  </button>
                </div>
              )
            })}
            {showAll &&
              companions.map((v) => (
                <div className="card row-card" key={v.id}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="truncate mono" style={{ fontSize: 12 }}>{v.id}</div>
                    <div className="faint" style={{ fontSize: 11 }}>
                      {fmtBytes(v.bytes)} ·{' '}
                      {v.kind === 'mmproj'
                        ? 'vision projector — fetched automatically with a model'
                        : 'separate MTP module — not a standalone model, and the bundled llama.cpp cannot attach one'}
                    </div>
                  </div>
                  {v.kind === 'mmproj' && <button onClick={() => void download(v.id)}>Download</button>}
                </div>
              ))}
            {!files.length && <div className="empty">Loading files…</div>}
          </div>

          <p className="faint" style={{ fontSize: 11, marginTop: 10 }}>
            A model split into parts downloads as one: every part, in one go. Multimodal models need their mmproj
            companion — it is fetched automatically alongside the model.
          </p>
        </>
      )}
    </>
  )
}
