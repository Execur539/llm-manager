import { useEffect, useState } from 'react'
import { invoke, on, fmtDuration } from '../lib/api'

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
      <h1>API server</h1>
      <p className="subtitle">
        Serves an OpenAI-compatible API, plus an Anthropic Messages endpoint — so Claude-compatible clients can
        use a local model too.
      </p>

      {error && (
        <div className="card" style={{ borderColor: '#5c2626' }}>
          <span className="badge bad">error</span> {error}
        </div>
      )}

      <div className="card">
        <div className="card-title">
          Status
          <span className={`badge ${status.running ? 'good' : ''}`}>{status.running ? 'running' : 'stopped'}</span>
        </div>

        <dl className="kv">
          <dt>Port</dt>
          <dd>
            <input
              type="number"
              className="narrow"
              value={port}
              disabled={status.running}
              onChange={(e) => setPort(Number(e.target.value))}
              onBlur={async () => {
                const s = await invoke<{ server: Record<string, unknown> }>('settings:get')
                await invoke('settings:patch', { server: { ...s.server, port } })
              }}
            />
          </dd>
          <dt>Base URL</dt>
          <dd>
            <span className="mono">{baseUrl}</span>
            <button
              style={{ marginLeft: 8 }}
              onClick={() => {
                void navigator.clipboard.writeText(baseUrl)
                setCopied(true)
                setTimeout(() => setCopied(false), 1500)
              }}
            >
              {copied ? 'Copied' : 'Copy'}
            </button>
          </dd>
          <dt>Auto-load models</dt>
          <dd>
            <input
              type="checkbox"
              checked={jitLoad}
              onChange={async (e) => {
                setJitLoad(e.target.checked)
                const s = await invoke<{ server: Record<string, unknown> }>('settings:get')
                await invoke('settings:patch', { server: { ...s.server, jitLoad: e.target.checked } })
              }}
            />{' '}
            <span className="faint">load whichever model a request names</span>
          </dd>
          <dt>API key</dt>
          <dd>
            {apiKey ? (
              <span className="mono">{apiKey}</span>
            ) : status.hasApiKey ? (
              <span className="faint">set — generating a new one replaces it</span>
            ) : (
              <span className="faint">none set</span>
            )}
            <button
              style={{ marginLeft: 8 }}
              onClick={async () => {
                setApiKey(await invoke<string>('server:generate-key'))
                setStatus(await invoke<ServerStatus>('server:status'))
              }}
            >
              {status.hasApiKey ? 'Replace' : 'Generate'}
            </button>
            {(apiKey || status.hasApiKey) && (
              <button
                style={{ marginLeft: 6 }}
                onClick={async () => {
                  await invoke('server:clear-key')
                  setApiKey(null)
                  setStatus(await invoke<ServerStatus>('server:status'))
                }}
              >
                Clear
              </button>
            )}
          </dd>
          <dt>Queue</dt>
          <dd>
            {status.queue ? `${status.queue} waiting` : <span className="faint">idle</span>}
          </dd>
        </dl>

        {status.networkBlocked && (
          <div className="card note" style={{ marginTop: 12 }}>
            Remote access is on, but the API has no key — so it is listening on 127.0.0.1 only
            rather than publishing an unauthenticated endpoint to your network. Generate a key
            above and restart the server to reach it from another machine.
          </div>
        )}

        <button className={status.running ? 'danger' : 'primary'} style={{ marginTop: 12 }} onClick={() => void toggle()}>
          {status.running ? 'Stop server' : 'Start server'}
        </button>
      </div>

      <div className="card">
        <div className="card-title">Endpoints</div>
        <div className="mono" style={{ fontSize: 11, lineHeight: 1.9 }}>
          GET&nbsp;&nbsp;/v1/models<br />
          POST /v1/chat/completions&nbsp;&nbsp;<span className="faint">— OpenAI, streaming and tools</span><br />
          POST /v1/embeddings<br />
          POST /v1/messages&nbsp;&nbsp;<span className="faint">— Anthropic Messages, streaming</span><br />
          GET&nbsp;&nbsp;/health
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          Request log
          <span className="faint" style={{ fontSize: 11 }}>local requests take priority over remote</span>
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
