import { useCallback, useEffect, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from 'react'
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

/** The sidebar's width range when expanded, matching Unsloth Studio's. */
const SIDEBAR_MIN = 260
const SIDEBAR_MAX = 480
const SIDEBAR_DEFAULT = 280

const clampSidebar = (px: number): number => Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, Math.round(px)))

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
  /*
   * The sidebar collapsed to icons, remembered between launches.
   *
   * Storage can throw (a remote tab with site data blocked), and a collapse preference is not worth
   * failing to render over, so both reads and writes fall back quietly.
   */
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    try {
      return localStorage.getItem('llmm.sidebarCollapsed') === '1'
    } catch {
      return false
    }
  })
  useEffect(() => {
    try {
      localStorage.setItem('llmm.sidebarCollapsed', collapsed ? '1' : '0')
    } catch {
      /* remembered for this session only */
    }
  }, [collapsed])

  /* The expanded width, dragged from the sidebar's edge and remembered the same way. */
  const [sidebarWidth, setSidebarWidth] = useState<number>(() => {
    try {
      const stored = Number(localStorage.getItem('llmm.sidebarWidth'))
      return stored ? clampSidebar(stored) : SIDEBAR_DEFAULT
    } catch {
      return SIDEBAR_DEFAULT
    }
  })
  useEffect(() => {
    // Written once the width settles rather than on every frame of a drag.
    const timer = setTimeout(() => {
      try {
        localStorage.setItem('llmm.sidebarWidth', String(sidebarWidth))
      } catch {
        /* remembered for this session only */
      }
    }, 250)
    return () => clearTimeout(timer)
  }, [sidebarWidth])

  /*
   * Follow the pointer from the edge, one layout per animation frame.
   *
   * Moving the edge reflows the whole main pane, so updating on every pointer event — several per
   * frame on a high-rate mouse — would lay the page out more often than it can be painted.
   * Capturing the pointer keeps the drag alive when it runs ahead of the handle.
   */
  const startResize = (e: ReactPointerEvent<HTMLDivElement>): void => {
    if (e.button !== 0) return
    e.preventDefault()
    const handle = e.currentTarget
    handle.setPointerCapture(e.pointerId)
    const startX = e.clientX
    const startWidth = sidebarWidth
    let next = startWidth
    let frame = 0
    document.body.classList.add('resizing-sidebar')
    const move = (ev: PointerEvent): void => {
      next = clampSidebar(startWidth + ev.clientX - startX)
      if (!frame) {
        frame = requestAnimationFrame(() => {
          frame = 0
          setSidebarWidth(next)
        })
      }
    }
    const end = (): void => {
      handle.removeEventListener('pointermove', move)
      handle.removeEventListener('pointerup', end)
      handle.removeEventListener('pointercancel', end)
      if (frame) cancelAnimationFrame(frame)
      setSidebarWidth(next)
      document.body.classList.remove('resizing-sidebar')
    }
    handle.addEventListener('pointermove', move)
    handle.addEventListener('pointerup', end)
    handle.addEventListener('pointercancel', end)
  }

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
    <div
      className={`app${collapsed ? ' sidebar-collapsed' : ''}`}
      // Only while expanded: an inline width would override the collapsed rail's.
      style={collapsed ? undefined : ({ '--sidebar-width': `${sidebarWidth}px` } as CSSProperties)}
    >
      <nav className="sidebar">
        <div className="brand">
          <BrandMark />
          <span className="brand-name">LLM Manager</span>
          {!isDesktop && <span className="badge">remote</span>}
          <button
            type="button"
            className="sidebar-toggle"
            onClick={() => setCollapsed((c) => !c)}
            title={collapsed ? 'Expand the sidebar' : 'Collapse the sidebar'}
            aria-label={collapsed ? 'Expand the sidebar' : 'Collapse the sidebar'}
            aria-expanded={!collapsed}
            data-testid="sidebar-toggle"
          >
            <Icon name="sidebar" size={15} />
          </button>
        </div>

        {groups.map((group) => (
          <div key={group} className="nav-section">
            <div className="nav-group">{group}</div>
            {NAV.filter((n) => n.group === group).map((n) => (
              <button
                key={n.id}
                type="button"
                className={`nav-item ${view === n.id ? 'active' : ''}`}
                onClick={() => setView(n.id)}
                // With the labels hidden, the name has to be reachable some other way.
                title={collapsed ? n.label : undefined}
                aria-current={view === n.id ? 'page' : undefined}
              >
                <Icon name={n.icon} />
                <span>{n.label}</span>
              </button>
            ))}
          </div>
        ))}

        <div
          className="sidebar-footer"
          title={collapsed ? (loaded ? `Loaded: ${loaded.model}` : 'No model loaded') : undefined}
        >
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

      {!collapsed && (
        <div
          className="sidebar-resizer"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize the sidebar"
          aria-valuemin={SIDEBAR_MIN}
          aria-valuemax={SIDEBAR_MAX}
          aria-valuenow={sidebarWidth}
          tabIndex={0}
          title="Drag to resize · double-click to reset"
          onPointerDown={startResize}
          onDoubleClick={() => setSidebarWidth(SIDEBAR_DEFAULT)}
          onKeyDown={(e) => {
            const step = e.shiftKey ? 40 : 10
            if (e.key === 'ArrowLeft') setSidebarWidth((w) => clampSidebar(w - step))
            else if (e.key === 'ArrowRight') setSidebarWidth((w) => clampSidebar(w + step))
            else if (e.key === 'Home') setSidebarWidth(SIDEBAR_MIN)
            else if (e.key === 'End') setSidebarWidth(SIDEBAR_MAX)
            else return
            e.preventDefault()
          }}
          data-testid="sidebar-resizer"
        />
      )}

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
        {view === 'library' && (
          <Library
            models={models}
            onRefresh={refreshModels}
            onLoaded={refreshLoaded}
            detection={hardware?.detection?.state ?? null}
          />
        )}
        {view === 'discover' && <Discover onDownloaded={refreshModels} />}
        {view === 'chat' && <ChatView loaded={loaded} />}
        {view === 'agent' && <AgentView loaded={loaded} />}
        {view === 'documents' && <Documents />}
        {view === 'server' && <ServerView />}
        {view === 'remote' && <RemoteView />}
        {view === 'settings' && <Settings models={models} onModelsChanged={refreshModels} />}
        </div>
      </main>

      <PermissionPrompt />
      <QuestionPrompt />
      <Toasts />
    </div>
  )
}
