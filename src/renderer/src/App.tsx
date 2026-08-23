import { useCallback, useEffect, useState } from 'react'
import type { HardwareSnapshot, ModelRecord } from '@shared/types'
import { invoke, on, isDesktop } from './lib/api'
import Dashboard from './views/Dashboard'
import Library from './views/Library'
import Discover from './views/Discover'
import ChatView from './views/Chat'
import AgentView from './views/Agent'
import Documents from './views/Documents'
import ServerView from './views/Server'
import RemoteView from './views/Remote'
import Settings from './views/Settings'
import PermissionPrompt from './components/PermissionPrompt'

export type View =
  | 'dashboard'
  | 'library'
  | 'discover'
  | 'chat'
  | 'agent'
  | 'documents'
  | 'server'
  | 'remote'
  | 'settings'

const NAV: { id: View; label: string; group: string }[] = [
  { id: 'dashboard', label: 'Dashboard', group: 'Overview' },
  { id: 'chat', label: 'Chat', group: 'Use' },
  { id: 'agent', label: 'Agent', group: 'Use' },
  { id: 'documents', label: 'Documents', group: 'Use' },
  { id: 'library', label: 'My models', group: 'Models' },
  { id: 'discover', label: 'Find a model', group: 'Models' },
  { id: 'server', label: 'API server', group: 'Serve' },
  { id: 'remote', label: 'Remote access', group: 'Serve' },
  { id: 'settings', label: 'Settings', group: 'Serve' }
]

export interface LoadedModel {
  model: string
  modelId: string
  port: number
  plan: { contextLength: number; kvType: string; gpuLayers: number; totalLayers: number }
}

export default function App(): JSX.Element {
  const [view, setView] = useState<View>('dashboard')
  const [hardware, setHardware] = useState<HardwareSnapshot | null>(null)
  const [models, setModels] = useState<ModelRecord[]>([])
  const [loaded, setLoaded] = useState<LoadedModel | null>(null)
  const [banner, setBanner] = useState<string | null>(null)

  const refreshModels = useCallback(async () => {
    setModels(await invoke<ModelRecord[]>('library:scan'))
  }, [])

  const refreshLoaded = useCallback(async () => {
    setLoaded(await invoke<LoadedModel | null>('model:status'))
  }, [])

  useEffect(() => {
    void (async () => {
      try {
        setHardware(await invoke<HardwareSnapshot>('hardware:get'))
        setModels(await invoke<ModelRecord[]>('library:scan'))
        setLoaded(await invoke<LoadedModel | null>('model:status'))
      } catch (err) {
        setBanner(err instanceof Error ? err.message : String(err))
      }
    })()

    const offs = [
      on<HardwareSnapshot>('hardware:update', setHardware),
      on<ModelRecord[]>('library:update', setModels),
      on<LoadedModel | null>('model:status', setLoaded),
      on<{ suggestion: string | null }>('autofit:verified', (v) => {
        if (v.suggestion) setBanner(v.suggestion)
      })
    ]
    return () => offs.forEach((off) => off())
  }, [])

  const groups = [...new Set(NAV.map((n) => n.group))]

  return (
    <div className="app">
      <nav className="sidebar">
        <div className="brand">
          LLM Manager
          {!isDesktop && <span className="badge" style={{ marginLeft: 8 }}>remote</span>}
        </div>

        {groups.map((group) => (
          <div key={group}>
            <div className="nav-group">{group}</div>
            {NAV.filter((n) => n.group === group).map((n) => (
              <div
                key={n.id}
                className={`nav-item ${view === n.id ? 'active' : ''}`}
                onClick={() => setView(n.id)}
              >
                {n.label}
              </div>
            ))}
          </div>
        ))}

        <div className="sidebar-footer">
          {loaded ? (
            <>
              <div className="badge good">loaded</div>
              <div className="truncate" style={{ marginTop: 6 }} title={loaded.model}>
                {loaded.model}
              </div>
              <div className="faint">
                {loaded.plan.contextLength.toLocaleString()} ctx · {loaded.plan.gpuLayers}/
                {loaded.plan.totalLayers} on GPU
              </div>
              <button
                style={{ marginTop: 8, width: '100%' }}
                onClick={async () => {
                  await invoke('model:unload')
                  await refreshLoaded()
                }}
              >
                Unload
              </button>
            </>
          ) : (
            <div className="faint">No model loaded</div>
          )}
        </div>
      </nav>

      <main className="main">
        {banner && (
          <div className="card" style={{ borderColor: 'var(--accent-dim)' }}>
            <div className="row">
              <span className="badge">note</span>
              <span style={{ flex: 1 }}>{banner}</span>
              <button onClick={() => setBanner(null)}>Dismiss</button>
            </div>
          </div>
        )}

        {view === 'dashboard' && <Dashboard hardware={hardware} models={models} loaded={loaded} onNavigate={setView} />}
        {view === 'library' && <Library models={models} onRefresh={refreshModels} onLoaded={refreshLoaded} />}
        {view === 'discover' && <Discover onDownloaded={refreshModels} />}
        {view === 'chat' && <ChatView loaded={loaded} />}
        {view === 'agent' && <AgentView loaded={loaded} />}
        {view === 'documents' && <Documents />}
        {view === 'server' && <ServerView />}
        {view === 'remote' && <RemoteView />}
        {view === 'settings' && <Settings />}
      </main>

      <PermissionPrompt />
    </div>
  )
}
