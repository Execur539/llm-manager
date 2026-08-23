import { useEffect, useState } from 'react'
import type { HardwareSnapshot, ModelRecord } from '@shared/types'
import Dashboard from './views/Dashboard'
import Library from './views/Library'
import AgentView from './views/Agent'
import Settings from './views/Settings'
import PermissionPrompt from './components/PermissionPrompt'

type View = 'dashboard' | 'library' | 'agent' | 'settings'

const NAV: { id: View; label: string }[] = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'library', label: 'Models' },
  { id: 'agent', label: 'Agent' },
  { id: 'settings', label: 'Settings' }
]

export default function App(): JSX.Element {
  const [view, setView] = useState<View>('dashboard')
  const [hardware, setHardware] = useState<HardwareSnapshot | null>(null)
  const [models, setModels] = useState<ModelRecord[]>([])
  const [loaded, setLoaded] = useState<{ model: string; port: number } | null>(null)

  useEffect(() => {
    void (async () => {
      setHardware(await window.api.hardware.get())
      setModels(await window.api.library.scan())
      setLoaded(await window.api.model.status())
    })()
    window.api.hardware.onUpdate((hw) => setHardware(hw as HardwareSnapshot))
  }, [])

  const refreshModels = async (): Promise<void> => {
    setModels(await window.api.library.scan())
  }

  const refreshLoaded = async (): Promise<void> => {
    setLoaded(await window.api.model.status())
  }

  return (
    <div className="app">
      <nav className="sidebar">
        <div className="brand">LLM Manager</div>
        {NAV.map((n) => (
          <div
            key={n.id}
            className={`nav-item ${view === n.id ? 'active' : ''}`}
            onClick={() => setView(n.id)}
          >
            {n.label}
          </div>
        ))}
        <div className="sidebar-footer">
          {loaded ? (
            <>
              <div className="badge good">loaded</div>
              <div style={{ marginTop: 6 }}>{loaded.model}</div>
              <div className="faint">127.0.0.1:{loaded.port}</div>
            </>
          ) : (
            <div className="faint">No model loaded</div>
          )}
        </div>
      </nav>

      <main className="main">
        {view === 'dashboard' && <Dashboard hardware={hardware} models={models} loaded={loaded} />}
        {view === 'library' && (
          <Library models={models} onRefresh={refreshModels} onLoaded={refreshLoaded} />
        )}
        {view === 'agent' && <AgentView modelLoaded={!!loaded} />}
        {view === 'settings' && <Settings />}
      </main>

      <PermissionPrompt />
    </div>
  )
}
