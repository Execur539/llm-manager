import { useEffect, useState } from 'react'
import { invoke, on, fmtDuration } from '../lib/api'
import NumberField from '../components/NumberField'
import { Toolbar } from '../lib/shell'

interface ServerStatus {
  running: boolean
  port: number | null
  /** Requests waiting on the one inference slot, including the one in flight. */
  queue?: number
  /** Whether a key exists. Never the key itself — this status is readable remotely. */
  hasApiKey?: boolean
  /** True when network exposure was asked for and refused because no key is set. */
  networkBlocked?: boolean
}

interface RequestEntry {
  id: number
  ts: number
  endpoint: string
  modelId: string | null
  tokensOut: number
  ms: number
  client: string
  status: number
}

/** What the server answers, in the order a client usually meets them. */
const ENDPOINTS: { method: 'GET' | 'POST'; path: string; note?: string }[] = [
  { method: 'GET', path: '/v1/models' },
  { method: 'POST', path: '/v1/chat/completions', note: 'OpenAI, streaming and tools' },
  { method: 'POST', path: '/v1/embeddings' },
  { method: 'POST', path: '/v1/messages', note: 'Anthropic Messages, streaming' },
  { method: 'GET', path: '/health' }
]

export default function ServerView(): JSX.Element {
  const [status, setStatus] = useState<ServerStatus>({ running: false, port: null })
  const [requests, setRequests] = useState<RequestEntry[]>([])
  /*
   * The key as generated in this session, so it can be read once and copied.
   *
   * Whether a key *exists* comes from the status instead. Nothing used to ask, so the panel said
   * "none set" after every restart and the natural response — press Generate — replaced a key
   * that every configured client was still using.
   */
  const [apiKey, setApiKey] = useState<string | null>(null)
  const [port, setPort] = useState(1234)
  const [jitLoad, setJitLoad] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    void invoke<ServerStatus>('server:status').then(setStatus).catch(() => undefined)
    void invoke<{ server: { port: number; jitLoad: boolean } }>('settings:get').then((s) => {
      setPort(s.server.port)
      setJitLoad(s.server.jitLoad)
    })

    const tick = (): void => {
      void invoke<RequestEntry[]>('server:requests', 50).then(setRequests).catch(() => undefined)
      // Queue depth and key state move without any event to announce them.
      void invoke<ServerStatus>('server:status').then(setStatus).catch(() => undefined)
    }
    tick()
    const timer = setInterval(tick, 3000)
    const off = on<ServerStatus>('server:status', setStatus)
    return () => {
      clearInterval(timer)
      off()
    }
  }, [])

  const toggle = async (): Promise<void> => {
    setError(null)
    try {
      if (status.running) setStatus(await invoke<ServerStatus>('server:stop'))
      else setStatus(await invoke<ServerStatus>('server:start'))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const baseUrl = `http://127.0.0.1:${status.port ?? port}/v1`

  return (
    <>
      <Toolbar title="API server" />
      <p className="subtitle">
        Serves an OpenAI-compatible API, plus an Anthropic Messages endpoint — so Claude-compatible clients can
        use a local model too.
      </p>

      {error && (
        <div className="card error-card">
          <span className="badge bad">error</span> {error}
        </div>
      )}

      {/* Whether it is serving, and where, first; then the few things that shape it. */}
      <div className="card server-card">
        <div className="server-head">
          <div className="server-state">
            <span className="sr-only">Status</span>
            <span className={`server-dot${status.running ? ' on' : ''}`} aria-hidden="true" />
            <span className="server-state-text">{status.running ? 'Running' : 'Stopped'}</span>
          </div>
          <button className={status.running ? 'danger' : 'primary'} onClick={() => void toggle()}>
            {status.running ? 'Stop server' : 'Start server'}
          </button>
        </div>
        <div className="server-url">
          <code>{baseUrl}</code>
          <button
            className="ghost tiny"
            onClick={() => {
              void navigator.clipboard.writeText(baseUrl)
              setCopied(true)
              setTimeout(() => setCopied(false), 1500)
            }}
          >
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>

        <div className="settings-rows">
          <div className="setting-row">
            <div className="setting-text">
              <div className="setting-label">Port</div>
              <div className="setting-hint">Changed while the server is stopped</div>
            </div>
            <div className="setting-control">
              <NumberField
                value={port}
                min={1}
                max={65535}
                disabled={status.running}
                onCommit={async (n) => {
                  setPort(n)
                  const s = await invoke<{ server: Record<string, unknown> }>('settings:get')
                  await invoke('settings:patch', { server: { ...s.server, port: n } })
                }}
              />
            </div>
          </div>
          <div className="setting-row">
            <div className="setting-text">
              <div className="setting-label">Auto-load models</div>
              <div className="setting-hint">Load whichever model a request names</div>
            </div>
            <div className="setting-control">
              <input
                type="checkbox"
                checked={jitLoad}
                aria-label="Auto-load models"
                onChange={async (e) => {
                  setJitLoad(e.target.checked)
                  const s = await invoke<{ server: Record<string, unknown> }>('settings:get')
                  await invoke('settings:patch', { server: { ...s.server, jitLoad: e.target.checked } })
                }}
              />
            </div>
          </div>
          <div className="setting-row">
            <div className="setting-text">
              <div className="setting-label">API key</div>
              <div className="setting-hint">
                {apiKey
                  ? 'Copy it now: it is shown once'
                  : status.hasApiKey
                    ? 'Set. Generating a new one replaces it'
                    : 'None set'}
              </div>
            </div>
            <div className="setting-control">
              {apiKey && <code className="server-key">{apiKey}</code>}
              <button
                onClick={async () => {
                  setApiKey(await invoke<string>('server:generate-key'))
                  setStatus(await invoke<ServerStatus>('server:status'))
                }}
              >
                {status.hasApiKey ? 'Replace' : 'Generate'}
              </button>
              {(apiKey || status.hasApiKey) && (
                <button
                  className="ghost"
                  onClick={async () => {
                    await invoke('server:clear-key')
                    setApiKey(null)
                    setStatus(await invoke<ServerStatus>('server:status'))
                  }}
                >
                  Clear
                </button>
              )}
            </div>
          </div>
          <div className="setting-row">
            <div className="setting-text">
              <div className="setting-label">Queue</div>
            </div>
            <div className="setting-control">
              <span className="faint">{status.queue ? `${status.queue} waiting` : 'Idle'}</span>
            </div>
          </div>
        </div>

        {status.networkBlocked && (
          <div className="card note server-note">
            Remote access is on, but the API has no key, so it is listening on 127.0.0.1 only rather than publishing an
            unauthenticated endpoint to your network. Generate a key above and restart the server to reach it from
            another machine.
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-title">Endpoints</div>
        <div className="endpoint-list">
          {ENDPOINTS.map((e) => (
            <div className="endpoint" key={e.path}>
              <span className="endpoint-method">{e.method}</span>
              <code className="endpoint-path">{e.path}</code>
              {e.note && <span className="endpoint-note">{e.note}</span>}
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          Request log
          <span className="card-title-note">Local requests take priority over remote ones</span>
        </div>
        {requests.length === 0 && <div className="empty">No requests yet.</div>}
        {requests.length > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Endpoint</th>
                  <th>Model</th>
                  <th>Tokens</th>
                  <th>Duration</th>
                  <th>Client</th>
                </tr>
              </thead>
              <tbody>
                {requests.map((r) => (
                  <tr key={r.id}>
                    <td className="faint">{new Date(r.ts).toLocaleTimeString()}</td>
                    <td className="mono">{r.endpoint}</td>
                    <td className="truncate" style={{ maxWidth: 180 }}>{r.modelId ?? '—'}</td>
                    <td>{r.tokensOut || '—'}</td>
                    <td>{fmtDuration(r.ms)}</td>
                    <td>
                      <span className={`badge ${r.client === 'local' ? '' : 'warn'}`}>{r.client}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  )
}
