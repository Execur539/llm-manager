import { useEffect, useState } from 'react'
import type { AppSettings } from '@shared/types'

export default function Settings(): JSX.Element {
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [missing, setMissing] = useState<string[]>([])

  useEffect(() => {
    void (async () => {
      setSettings(await window.api.settings.get())
      setMissing((await window.api.runtime.missingBinaries()) as string[])
    })()
  }, [])

  if (!settings) return <div>Loading…</div>

  const patch = async (p: Partial<AppSettings>): Promise<void> => {
    setSettings(await window.api.settings.patch(p))
  }

  return (
    <>
      <h1>Settings</h1>
      <p className="subtitle">Stored in %APPDATA%\LLMManager — survives the app being moved.</p>

      {missing.length > 0 && (
        <div className="card" style={{ borderColor: '#5a4515' }}>
          <div className="card-title">
            Missing bundled binaries <span className="badge warn">setup required</span>
          </div>
          <div className="dim">
            These are not present yet, so the features that need them will not work:
            <div className="mono" style={{ marginTop: 6 }}>
              {missing.join(', ')}
            </div>
            <div style={{ marginTop: 8 }}>See BUILD_STATUS.md for how to fetch them.</div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-title">Auto-fit</div>
        <dl className="kv">
          <dt>Target context</dt>
          <dd>
            <input
              type="number"
              value={settings.autoFit.targetContext}
              onChange={(e) =>
                void patch({ autoFit: { ...settings.autoFit, targetContext: Number(e.target.value) } })
              }
            />
          </dd>
          <dt>Ideal context</dt>
          <dd>
            <input
              type="number"
              value={settings.autoFit.idealContext}
              onChange={(e) =>
                void patch({ autoFit: { ...settings.autoFit, idealContext: Number(e.target.value) } })
              }
            />
          </dd>
          <dt>Preferred KV</dt>
          <dd>{settings.autoFit.preferredKvType}</dd>
          <dt>KV floor</dt>
          <dd>{settings.autoFit.minKvType}</dd>
          <dt>Headroom (MB)</dt>
          <dd>
            <input
              type="number"
              value={settings.autoFit.headroomMb}
              onChange={(e) =>
                void patch({ autoFit: { ...settings.autoFit, headroomMb: Number(e.target.value) } })
              }
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
            />{' '}
            <span className="faint">read-only until a plan is approved</span>
          </dd>
          <dt>Max tool calls / turn</dt>
          <dd>
            <input
              type="number"
              value={settings.agent.maxToolCallsPerTurn}
              onChange={(e) =>
                void patch({ agent: { ...settings.agent, maxToolCallsPerTurn: Number(e.target.value) } })
              }
            />
          </dd>
          <dt>Command timeout (ms)</dt>
          <dd>
            <input
              type="number"
              value={settings.agent.commandTimeoutMs}
              onChange={(e) =>
                void patch({ agent: { ...settings.agent, commandTimeoutMs: Number(e.target.value) } })
              }
            />
          </dd>
          <dt>Hard blocks</dt>
          <dd>
            <span className={`badge ${settings.agent.hardBlocksDisabled ? 'bad' : 'good'}`}>
              {settings.agent.hardBlocksDisabled ? 'DISABLED' : 'enabled'}
            </span>
          </dd>
        </dl>
      </div>
    </>
  )
}
