import { useEffect, useState } from 'react'
import type { AppSettings } from '@shared/types'
import { invoke, on, fmtBytes } from '../lib/api'
import ConfirmDialog from '../components/ConfirmDialog'
import NumberField from '../components/NumberField'
import VideoSettings from '../components/VideoSettings'

interface McpStatus {
  id: string
  name: string
  connected: boolean
  tools: string[]
  error: string | null
}

interface MemoryEntry {
  id: string
  text: string
  updatedAt: number
}

export default function Settings(): JSX.Element {
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [missing, setMissing] = useState<string[]>([])
  const [paths, setPaths] = useState<{ exeDir: string; modelsDir: string } | null>(null)
  const [mcp, setMcp] = useState<McpStatus[]>([])
  const [memory, setMemory] = useState<MemoryEntry[]>([])
  const [hfToken, setHfToken] = useState('')
  const [update, setUpdate] = useState<{ available: boolean; latestVersion: string | null; notes: string | null; downloadUrl: string | null; bytes?: number; error?: string; previousFailure?: string } | null>(null)
  const [version, setVersion] = useState('')
  const [info, setInfo] = useState<string | null>(null)
  const [confirmHardBlocks, setConfirmHardBlocks] = useState(false)
  /** Bytes of the update fetched so far, or null when no download is running. */
  const [downloading, setDownloading] = useState<{ done: number; total: number } | null>(null)

  /*
   * The download reports progress; until now nobody listened.
   *
   * Subscribed for the life of the view rather than only while a download runs, so returning to
   * Settings mid-download picks the progress back up instead of showing an idle button for a
   * transfer that is still going.
   */
  useEffect(() => on<{ done: number; total: number }>('update:progress', setDownloading), [])

  // New MCP server form
  const [mcpName, setMcpName] = useState('')
  const [mcpCommand, setMcpCommand] = useState('')
  const [mcpArgs, setMcpArgs] = useState('')

  const refresh = async (): Promise<void> => {
    setSettings(await invoke<AppSettings>('settings:get'))
    setMissing(await invoke<string[]>('runtime:missing-binaries'))
    setPaths(await invoke('app:paths'))
    setMcp(await invoke<McpStatus[]>('mcp:list'))
    setMemory(await invoke<MemoryEntry[]>('agent:memory'))
    setVersion(await invoke<string>('app:version'))
  }

  useEffect(() => {
    void refresh()
    const off = on<McpStatus[]>('mcp:update', setMcp)
    return off
  }, [])

  if (!settings) return <div className="empty">Loading…</div>

  const patch = async (p: Partial<AppSettings>): Promise<void> => {
    setSettings(await invoke<AppSettings>('settings:patch', p))
  }

  return (
    <>
      <h1>Settings</h1>
      <p className="subtitle">
        Stored in %APPDATA%\LLMManager, so they survive the app being moved. No telemetry is collected, ever.
      </p>

      {info && <div className="card note">{info}</div>}

      {missing.length > 0 && (
        <div className="card" style={{ borderColor: '#5a4515' }}>
          <div className="card-title">
            Bundled components missing <span className="badge warn">{missing.length}</span>
          </div>
          <div className="dim">
            <div className="mono" style={{ margin: '6px 0' }}>{missing.join(', ')}</div>
            Run <code className="mono">npm run fetch-vendor</code> to download them. Features that need a missing
            component will say so rather than failing silently.
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-title">Auto-fit</div>
        <div className="dim" style={{ marginBottom: 10 }}>
          The engine maximises context, prefers a Q8 KV cache, and never drops below Q4. It sizes against
          <strong> free</strong> VRAM and splits multi-GPU work by real capacity.
        </div>
        <dl className="kv">
          <dt>Target context</dt>
          <dd>
            <NumberField
              value={settings.autoFit.targetContext}
              min={512}
              max={10_000_000}
              hint="minimum to aim for"
              onCommit={(n) => void patch({ autoFit: { ...settings.autoFit, targetContext: n } })}
            />
          </dd>
          <dt>Ideal context</dt>
          <dd>
            <NumberField
              value={settings.autoFit.idealContext}
              min={512}
              max={10_000_000}
              hint="stop growing here"
              onCommit={(n) => void patch({ autoFit: { ...settings.autoFit, idealContext: n } })}
            />
          </dd>
          <dt>Preferred KV</dt>
          <dd>
            <select
              className="kv-select"
              value={settings.autoFit.preferredKvType}
              onChange={(e) => void patch({ autoFit: { ...settings.autoFit, preferredKvType: e.target.value as 'f16' | 'q8_0' | 'q4_0' } })}
            >
              <option value="f16">f16 — best quality</option>
              <option value="q8_0">q8_0 — recommended</option>
              <option value="q4_0">q4_0 — smallest</option>
            </select>
            <span className="faint">aim for this first</span>
          </dd>
          <dt>KV floor</dt>
          <dd>
            <select
              className="kv-select"
              value={settings.autoFit.minKvType}
              onChange={(e) => void patch({ autoFit: { ...settings.autoFit, minKvType: e.target.value as 'f16' | 'q8_0' | 'q4_0' } })}
            >
              <option value="q4_0">q4_0 — smallest</option>
              <option value="q8_0">q8_0 — recommended</option>
              <option value="f16">f16 — best quality</option>
            </select>
            <span className="faint">never go below this</span>
          </dd>
          <dt>VRAM headroom</dt>
          <dd>
            <NumberField
              value={settings.autoFit.headroomMb}
              min={0}
              max={65_536}
              hint="MB left free for the desktop"
              onCommit={(n) => void patch({ autoFit: { ...settings.autoFit, headroomMb: n } })}
            />
          </dd>
        </dl>
      </div>

      <div className="card">
        <div className="card-title">Agent</div>
        <dl className="kv">
          <dt>Plan mode</dt>
          <dd>
            <input
              type="checkbox"
              checked={settings.agent.planMode}
              onChange={(e) => void patch({ agent: { ...settings.agent, planMode: e.target.checked } })}
            />
            <span className="faint" style={{ marginLeft: 6 }}>read-only until a plan is approved</span>
          </dd>
          <dt>Context strategy</dt>
          <dd>
            <select
              value={settings.agent.compaction}
              onChange={(e) => void patch({ agent: { ...settings.agent, compaction: e.target.value as 'auto-compact' | 'sliding-window' } })}
            >
              <option value="auto-compact">Auto-compact (summarise older turns)</option>
              <option value="sliding-window">Sliding window (drop oldest)</option>
            </select>
          </dd>
          <dt>Max tool calls / turn</dt>
          <dd>
            <NumberField
              value={settings.agent.maxToolCallsPerTurn}
              min={1}
              max={1000}
              hint="runaway-loop ceiling"
              onCommit={(n) => void patch({ agent: { ...settings.agent, maxToolCallsPerTurn: n } })}
            />
          </dd>
          <dt>Command timeout</dt>
          <dd>
            <NumberField
              value={settings.agent.commandTimeoutMs}
              min={1000}
              max={3_600_000}
              hint="ms"
              onCommit={(n) => void patch({ agent: { ...settings.agent, commandTimeoutMs: n } })}
            />
          </dd>
          <dt>Approvals</dt>
          <dd>
            <button onClick={async () => { await invoke('agent:clear-permission-rules'); setInfo('Remembered approvals cleared.') }}>
              Clear remembered approvals
            </button>
          </dd>
        </dl>

        <div className="danger-zone">
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <div>
              <strong>Hard blocks</strong>
              <div className="faint" style={{ fontSize: 11 }}>
                Refuses disk formatting, system-root deletion, bootloader writes and disabling security tools —
                in every permission mode.
              </div>
            </div>
            <span className={`badge ${settings.agent.hardBlocksDisabled ? 'bad' : 'good'}`}>
              {settings.agent.hardBlocksDisabled ? 'DISABLED' : 'enabled'}
            </span>
          </div>
          <button
            className="danger"
            style={{ marginTop: 10 }}
            data-testid="toggle-hard-blocks"
            onClick={() => {
              // Re-enabling protection is safe and immediate; removing it is not.
              if (settings.agent.hardBlocksDisabled) {
                void patch({ agent: { ...settings.agent, hardBlocksDisabled: false } })
              } else {
                setConfirmHardBlocks(true)
              }
            }}
          >
            {settings.agent.hardBlocksDisabled ? 'Re-enable hard blocks' : 'Disable hard blocks…'}
          </button>
        </div>
      </div>

      <div className="card">
        <div className="card-title">Video</div>
        {/*
          * A video is the one attachment that can fill a context window on its own, so what the
          * app does to it before the model sees it is worth being able to steer. Every control
          * here moves the token cost, and the panel prices them rather than describing them.
          */}
        <VideoSettings settings={settings} patch={patch} />
      </div>

      <div className="card">
        <div className="card-title">MCP servers</div>
        <div className="dim" style={{ marginBottom: 10 }}>
          Connect any Model Context Protocol server to add its tools to the agent. Servers run as local
          processes with your permissions, so only add ones you trust.
        </div>

        {mcp.map((s) => (
          <div className="row row-card" key={s.id}>
            <span className={`badge ${s.connected ? 'good' : 'bad'}`}>{s.connected ? 'connected' : 'offline'}</span>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="truncate">{s.name}</div>
              <div className="faint" style={{ fontSize: 11 }}>
                {s.connected ? `${s.tools.length} tools: ${s.tools.slice(0, 5).join(', ')}${s.tools.length > 5 ? '…' : ''}` : s.error ?? 'not connected'}
              </div>
            </div>
            <button onClick={async () => { await invoke('mcp:remove', s.id); await refresh() }}>Remove</button>
          </div>
        ))}
        {!mcp.length && <div className="empty" style={{ fontSize: 12 }}>No MCP servers configured.</div>}

        <div className="row" style={{ marginTop: 12, flexWrap: 'wrap' }}>
          <input type="text" placeholder="Name" value={mcpName} onChange={(e) => setMcpName(e.target.value)} style={{ width: 120 }} />
          <input type="text" placeholder="Command (e.g. npx)" value={mcpCommand} onChange={(e) => setMcpCommand(e.target.value)} style={{ width: 160 }} />
          <input type="text" placeholder="Args (space separated)" value={mcpArgs} onChange={(e) => setMcpArgs(e.target.value)} style={{ flex: 1, minWidth: 160 }} />
          <button
            onClick={async () => {
              if (!mcpName || !mcpCommand) return
              await invoke('mcp:add', {
                name: mcpName,
                transport: 'stdio',
                command: mcpCommand,
                args: mcpArgs.split(/\s+/).filter(Boolean),
                enabled: true
              })
              setMcpName('')
              setMcpCommand('')
              setMcpArgs('')
              await invoke('mcp:connect')
              await refresh()
            }}
          >
            Add
          </button>
          <button onClick={async () => { await invoke('mcp:connect'); await refresh() }}>Reconnect all</button>
        </div>
      </div>

      <div className="card">
        <div className="card-title">Agent memory</div>
        <div className="dim" style={{ marginBottom: 10 }}>
          What the agent has chosen to remember across sessions. This is the only context carried between
          sessions — there is no per-folder instructions file.
        </div>
        {memory.map((m) => (
          <div className="row row-card" key={m.id}>
            <div style={{ flex: 1, minWidth: 0 }}>{m.text}</div>
            <button onClick={async () => { await invoke('agent:memory-delete', m.id); await refresh() }}>Forget</button>
          </div>
        ))}
        {!memory.length && <div className="empty" style={{ fontSize: 12 }}>Nothing remembered yet.</div>}
      </div>

      <div className="card">
        <div className="card-title">HuggingFace</div>
        <div className="row">
          <input
            type="password"
            placeholder="Access token (for gated repos and higher rate limits)"
            value={hfToken}
            onChange={(e) => setHfToken(e.target.value)}
            style={{ flex: 1 }}
          />
          <button onClick={async () => { await invoke('settings:set-hf-token', hfToken || null); setHfToken(''); setInfo('Token saved.') }}>
            Save
          </button>
        </div>
      </div>

      <div className="card">
        <div className="card-title">Paths</div>
        <dl className="kv">
          <dt>App folder</dt>
          <dd className="truncate">{paths?.exeDir}</dd>
          <dt>Models folder</dt>
          <dd className="truncate">{paths?.modelsDir}</dd>
        </dl>
        <div className="faint" style={{ fontSize: 11, marginTop: 6 }}>
          Models live beside the app. If you move the app, it offers to bring them along on next launch.
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          About
          <span className="badge">v{version}</span>
        </div>
        <div className="row" style={{ flexWrap: 'wrap' }}>
          <button
            onClick={async () => {
              const result = await invoke<typeof update>('update:check')
              setUpdate(result)
              if (result && !result.available) setInfo(result.error ?? 'You are on the latest version.')
            }}
          >
            Check for updates
          </button>
          <button onClick={async () => { const p = await invoke<string>('diagnostics:reveal'); setInfo(`Diagnostics written to ${p}`) }}>
            Copy diagnostics
          </button>
        </div>

        {/*
          * Shown above the offer, not folded into `error`.
          *
          * `error` describes this check and is only rendered when nothing is available. A swap
          * that failed last time needs saying precisely when an update *is* on offer, because
          * that offer looks identical to the one that already failed once — and without this the
          * app would simply be repeating a claim it did not honour.
          */}
        {update?.previousFailure && (
          <div className="card" style={{ marginTop: 12, borderColor: '#5c2626' }}>
            <div className="card-title">
              The last update did not finish
              <span className="badge bad">not applied</span>
            </div>
            <div className="dim">{update.previousFailure}</div>
          </div>
        )}

        {update?.available && (
          <div className="card" style={{ marginTop: 12, borderColor: 'var(--accent-dim)' }}>
            <div className="card-title">Version {update.latestVersion} is available</div>
            <pre style={{ maxHeight: 160 }}>{update.notes}</pre>
            {/*
              * Nine hundred megabytes with nothing on screen is indistinguishable from a dead
              * button. The main process was already reporting progress and the channel was
              * already allowed through — nothing here had ever subscribed to it, so every byte
              * of it was computed, sent across the bridge and dropped. The first sign the click
              * had worked was the app restarting itself minutes later.
              */}
            {downloading ? (
              <div data-testid="update-progress">
                <div className="row" style={{ justifyContent: 'space-between', fontSize: 12 }}>
                  <span>Downloading version {update.latestVersion}…</span>
                  <span className="mono">
                    {downloading.total > 0 ? `${Math.round((downloading.done / downloading.total) * 100)}%` : fmtBytes(downloading.done)}
                  </span>
                </div>
                <div className="meter" style={{ marginTop: 6 }}>
                  <span style={{ width: downloading.total > 0 ? `${(downloading.done / downloading.total) * 100}%` : '100%' }} />
                </div>
                <div className="faint" style={{ fontSize: 11, marginTop: 6 }}>
                  {fmtBytes(downloading.done)}
                  {downloading.total > 0 && ` of ${fmtBytes(downloading.total)}`} · the app will restart itself to finish
                </div>
              </div>
            ) : (
              <button
                className="primary"
                onClick={async () => {
                  // Set before awaiting: the first progress frame can be a second or more away
                  // on a slow link, and the button must stop looking clickable immediately.
                  setDownloading({ done: 0, total: update.bytes ?? 0 })
                  const r = await invoke<{ ok: boolean; message: string }>('update:apply', update.downloadUrl)
                  if (!r?.ok) {
                    setDownloading(null)
                    setInfo(r?.message ?? 'The update could not be applied.')
                  }
                }}
                data-testid="update-apply"
              >
                Download and install
              </button>
            )}
          </div>
        )}
      </div>

      <ConfirmDialog
        open={confirmHardBlocks}
        danger
        title="Disable hard blocks?"
        confirmLabel="Disable hard blocks"
        requirePhrase="DISABLE HARD BLOCKS"
        body={
          <>
            <p>
              Hard blocks are the last thing standing between a confused model and an
              unrecoverable command. With them off the agent can format drives, delete system
              roots, overwrite the bootloader and disable security tooling.
            </p>
            <p className="faint">
              Approval prompts still apply. This only removes the refusals that no approval can
              override.
            </p>
          </>
        }
        onCancel={() => setConfirmHardBlocks(false)}
        onConfirm={() => {
          setConfirmHardBlocks(false)
          void patch({ agent: { ...settings.agent, hardBlocksDisabled: true } })
        }}
      />
    </>
  )
}
