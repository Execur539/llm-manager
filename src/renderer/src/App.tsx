import { useCallback, useEffect, useState } from 'react'
import type { HardwareSnapshot, ModelCapabilities, ModelRecord } from '@shared/types'
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
import Icon, { type IconName } from './components/Icon'
import BrandMark from './components/BrandMark'
import PermissionPrompt from './components/PermissionPrompt'
import QuestionPrompt from './components/QuestionPrompt'
import Toasts from './components/Toasts'

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

const NAV: { id: View; label: string; group: string; icon: IconName }[] = [
  { id: 'dashboard', label: 'Dashboard', group: 'Overview', icon: 'dashboard' },
  { id: 'chat', label: 'Chat', group: 'Use', icon: 'chat' },
  { id: 'agent', label: 'Agent', group: 'Use', icon: 'agent' },
  { id: 'documents', label: 'Documents', group: 'Use', icon: 'documents' },
  { id: 'library', label: 'My models', group: 'Models', icon: 'models' },
  { id: 'discover', label: 'Find a model', group: 'Models', icon: 'search' },
  { id: 'server', label: 'API server', group: 'Serve', icon: 'server' },
  { id: 'remote', label: 'Remote access', group: 'Serve', icon: 'remote' },
  { id: 'settings', label: 'Settings', group: 'Serve', icon: 'settings' }
]

/** Views whose layout owns the full height and scrolls internally. */
const FILL_VIEWS = new Set<View>(['chat', 'agent', 'documents'])

export interface LoadedModel {
  model: string
  modelId: string
  port: number
  plan: { contextLength: number; kvType: string; gpuLayers: number; totalLayers: number }
  /** Optional: a status emitted by an older build may not carry it. */
  caps?: ModelCapabilities
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
          <BrandMark />
          <span className="brand-name">LLM Manager</span>
          {!isDesktop && <span className="badge">remote</span>}
        </div>

        {groups.map((group) => (
          <div key={group}>
            <div className="nav-group">{group}</div>
            {NAV.filter((n) => n.group === group).map((n) => (
              <button
                key={n.id}
                type="button"
                className={`nav-item ${view === n.id ? 'active' : ''}`}
                onClick={() => setView(n.id)}
                aria-current={view === n.id ? 'page' : undefined}
              >
                <Icon name={n.icon} />
                <span>{n.label}</span>
              </button>
            ))}
          </div>
        ))}

        <div className="sidebar-footer">
          {loaded ? (
            <>
              <div className="loaded-head" data-testid="model-loaded">
                <span className="pulse-dot" aria-hidden="true" />
                <span className="loaded-label">Loaded</span>
              </div>
              <div className="loaded-name truncate" title={loaded.model}>
                {loaded.model}
              </div>
              <div className="loaded-meta">
                {loaded.plan.contextLength.toLocaleString()} ctx · {loaded.plan.gpuLayers}/
                {loaded.plan.totalLayers} layers on GPU
              </div>
              <button
                className="full"
                onClick={async () => {
                  await invoke('model:unload')
                  await refreshLoaded()
                }}
              >
                Unload
              </button>
            </>
          ) : (
            <div className="loaded-empty">
              <Icon name="chip" size={14} />
              <span>No model loaded</span>
            </div>
          )}
        </div>
      </nav>

      {/* Chat-like views fill the pane; the rest scroll normally. */}
      <main className={`main${FILL_VIEWS.has(view) ? ' fill' : ''}`}>
        {banner && (
          <div className="app-banner" role="status">
            <span className="badge">note</span>
            <span className="app-banner-text">{banner}</span>
            <button className="tiny" onClick={() => setBanner(null)} aria-label="Dismiss">
              Dismiss
            </button>
          </div>
        )}

        <div className="page">
        {view === 'dashboard' && <Dashboard hardware={hardware} models={models} loaded={loaded} onNavigate={setView} />}
        {view === 'library' && <Library models={models} onRefresh={refreshModels} onLoaded={refreshLoaded} />}
        {view === 'discover' && <Discover onDownloaded={refreshModels} />}
        {view === 'chat' && <ChatView loaded={loaded} />}
        {view === 'agent' && <AgentView loaded={loaded} />}
        {view === 'documents' && <Documents />}
        {view === 'server' && <ServerView />}
        {view === 'remote' && <RemoteView />}
        {view === 'settings' && <Settings />}
        </div>
      </main>

      <PermissionPrompt />
      <QuestionPrompt />
      <Toasts />
    </div>
  )
}
