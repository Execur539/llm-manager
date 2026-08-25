import { useEffect, useState } from 'react'
import { invoke, on } from '../lib/api'
import ConfirmDialog from '../components/ConfirmDialog'

interface RemoteStatus {
  tunnel: { running: boolean; url: string | null; error: string | null }
  web: boolean
  hasPassword: boolean
  settings: { enabled: boolean; mode: 'tunnel' | 'own-domain'; domain: string | null }
}

export default function RemoteView(): JSX.Element {
  const [status, setStatus] = useState<RemoteStatus | null>(null)
  const [password, setPassword] = useState('')
  const [domain, setDomain] = useState('')
  const [email, setEmail] = useState('')
  const [freednsToken, setFreednsToken] = useState('')
  const [mode, setMode] = useState<'tunnel' | 'own-domain'>('tunnel')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)
  const [url, setUrl] = useState<string | null>(null)
  const [remoteTools, setRemoteTools] = useState(false)
  const [confirmRemoteTools, setConfirmRemoteTools] = useState(false)

  const refresh = async (): Promise<void> => {
    const s = await invoke<RemoteStatus>('remote:status')
    setStatus(s)
    setMode(s.settings.mode)
    setDomain(s.settings.domain ?? '')
    if (s.tunnel.url) setUrl(s.tunnel.url)
  }

  useEffect(() => {
    void refresh()
    void invoke<{ agent: { remoteToolsEnabled: boolean } }>('settings:get').then((s) =>
      setRemoteTools(s.agent.remoteToolsEnabled)
    )
    const off = on<{ url: string | null }>('remote:status', (s) => {
      setUrl(s.url)
      void refresh()
    })
    return off
  }, [])

  const savePassword = async (): Promise<void> => {
    setError(null)
    try {
      await invoke('settings:set-password', password)
      setPassword('')
      setInfo('Password set. Any existing remote sessions were signed out.')
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const enable = async (): Promise<void> => {
    setBusy(true)
    setError(null)
    setInfo(null)
    try {
      const result = await invoke<{ url: string | null }>('remote:enable', mode, domain || undefined, email || undefined)
      setUrl(result.url)
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const disable = async (): Promise<void> => {
    setBusy(true)
    try {
      await invoke('remote:disable')
      setUrl(null)
      await refresh()
    } finally {
      setBusy(false)
    }
  }

  const active = status?.settings.enabled && (status.tunnel.running || status.web)

  const applyRemoteTools = async (enabled: boolean): Promise<void> => {
    setRemoteTools(enabled)
    const current = await invoke<{ agent: Record<string, unknown> }>('settings:get')
    await invoke('settings:patch', { agent: { ...current.agent, remoteToolsEnabled: enabled } })
  }

  return (
    <>
      <h1>Remote access</h1>
      <p className="subtitle">
        Reach this machine's models from a browser anywhere. Everything sits behind a password, and the
        inference server itself is never exposed directly.
      </p>

      {error && (
        <div className="card" style={{ borderColor: '#5c2626' }}>
          <span className="badge bad">error</span> {error}
        </div>
      )}
      {info && <div className="card note">{info}</div>}

      <div className="card">
        <div className="card-title">
          1. Password
          <span className={`badge ${status?.hasPassword ? 'good' : 'bad'}`}>
            {status?.hasPassword ? 'set' : 'required'}
          </span>
        </div>
        <div className="dim" style={{ marginBottom: 10 }}>
          Remote access cannot be enabled until a password exists. Minimum 10 characters — this is reachable
          from the internet.
        </div>
        <div className="row">
          <input
            type="password"
            placeholder="New password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={{ flex: 1 }}
          />
          <button onClick={() => void savePassword()} disabled={password.length < 10}>
            {status?.hasPassword ? 'Change' : 'Set'} password
          </button>
        </div>
      </div>

      <div className="card">
        <div className="card-title">2. How to connect</div>

        <label className="radio">
          <input type="radio" checked={mode === 'tunnel'} onChange={() => setMode('tunnel')} />
          <div>
            <strong>Tunnel</strong> <span className="badge good">easiest</span>
            <div className="faint">
              A public HTTPS address via Cloudflare. No router changes, works behind CGNAT, TLS handled for you.
              The address changes each time you enable it.
            </div>
          </div>
        </label>

        <label className="radio">
          <input type="radio" checked={mode === 'own-domain'} onChange={() => setMode('own-domain')} />
          <div>
            <strong>Your own domain</strong>
            <div className="faint">
              Points a domain at your home IP. Needs dynamic DNS, port forwarding (80 and 443), and issues a real
              Let's Encrypt certificate. Get a free hostname from{' '}
              <a href="https://freedns.afraid.org" target="_blank" rel="noreferrer">freedns.afraid.org</a>.
            </div>
          </div>
        </label>

        {mode === 'own-domain' && (
          <div className="indent">
            <dl className="kv">
              <dt>Domain</dt>
              <dd>
                <input type="text" placeholder="llm.example.mooo.com" value={domain} onChange={(e) => setDomain(e.target.value)} />
                <button
                  style={{ marginLeft: 6 }}
                  onClick={async () => {
                    const check = await invoke<{ ok: boolean; message: string }>('remote:check-domain', domain)
                    if (check.ok) setInfo(check.message)
                    else setError(check.message)
                  }}
                >
                  Check
                </button>
              </dd>
              <dt>Email</dt>
              <dd>
                <input type="text" placeholder="you@example.com" value={email} onChange={(e) => setEmail(e.target.value)} />
                <span className="faint" style={{ marginLeft: 6 }}>for Let's Encrypt expiry notices</span>
              </dd>
              <dt>FreeDNS token</dt>
              <dd>
                <input
                  type="text"
                  placeholder="Direct URL token from FreeDNS"
                  value={freednsToken}
                  onChange={(e) => setFreednsToken(e.target.value)}
                />
                <button
                  style={{ marginLeft: 6 }}
                  onClick={async () => {
                    await invoke('settings:set-freedns-token', freednsToken)
                    const r = await invoke<{ ok: boolean; message: string }>('remote:ddns-update')
                    if (r.ok) setInfo(r.message)
                    else setError(r.message)
                  }}
                >
                  Save & update
                </button>
              </dd>
            </dl>
            <div className="faint" style={{ fontSize: 11 }}>
              Certificates renew automatically in-process while the app runs — no scheduled task, so moving the
              app never breaks renewal.
            </div>
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-title">
          3. Agent tools over the internet
          <span className={`badge ${remoteTools ? 'bad' : 'good'}`}>{remoteTools ? 'ENABLED' : 'off'}</span>
        </div>
        <div className="dim">
          With this off, remote sessions can chat and read but cannot write files or run commands. Turning it on
          means anyone who gets past the password can run code on this machine.
        </div>
        <label className="row" style={{ marginTop: 10, gap: 8 }}>
          <input
            type="checkbox"
            checked={remoteTools}
            onChange={async (e) => {
              if (e.target.checked && !confirm('Enable full tool access for remote sessions?\n\nA leaked password would then allow arbitrary commands on this machine.')) return
              setRemoteTools(e.target.checked)
              const s = await invoke<{ agent: Record<string, unknown> }>('settings:get')
              await invoke('settings:patch', { agent: { ...s.agent, remoteToolsEnabled: e.target.checked } })
            }}
          />
          <span>Allow remote sessions to use write and execute tools</span>
        </label>
      </div>

      <div className="card">
        <div className="card-title">
          4. Connection
          <span className={`badge ${active ? 'good' : ''}`}>{active ? 'live' : 'off'}</span>
        </div>

        {url && (
          <div className="row" style={{ marginBottom: 10 }}>
            <span className="mono" style={{ flex: 1 }}>{url}</span>
            <button onClick={() => void navigator.clipboard.writeText(url)}>Copy</button>
          </div>
        )}

        {status?.tunnel.error && <div className="badge bad">{status.tunnel.error}</div>}

        <button
          className={active ? 'danger' : 'primary'}
          disabled={busy || !status?.hasPassword}
          onClick={() => void (active ? disable() : enable())}
        >
          {busy ? 'Working…' : active ? 'Disable remote access' : 'Enable remote access'}
        </button>

        {!status?.hasPassword && (
          <div className="faint" style={{ marginTop: 8, fontSize: 11 }}>Set a password first.</div>
        )}
      </div>

      <ConfirmDialog
        open={confirmRemoteTools}
        danger
        title="Allow remote sessions to run code?"
        confirmLabel="Enable full tool access"
        requirePhrase="enable full access"
        body={
          <>
            <p>
              Remote sessions would gain the write and execute tools: creating and deleting files, and running
              commands, on <strong>this machine</strong>.
            </p>
            <p className="subtitle">
              Anyone who gets past the remote password would have that same reach. Permission prompts still
              apply, but they would be answered on whichever device is connected.
            </p>
          </>
        }
        onCancel={() => setConfirmRemoteTools(false)}
        onConfirm={() => {
          setConfirmRemoteTools(false)
          void applyRemoteTools(true)
        }}
      />
    </>
  )
}
