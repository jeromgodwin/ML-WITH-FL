import React, { useEffect, useState } from 'react'
import Dashboard from './components/Dashboard'
import LiveScan from './components/LiveScan'
import FederatedLearning from './components/FederatedLearning'
import Security from './components/Security'
import System from './components/System'

const TABS = [
  { id: 'Dashboard', label: 'Dashboard', ico: '◧' },
  { id: 'Live Scan', label: 'Live Scan', ico: '⬢' },
  { id: 'Federated', label: 'Federated', ico: '⬡' },
  { id: 'Security', label: 'Security', ico: '⬔' },
  { id: 'System', label: 'System', ico: '⬣' },
]

const COMPONENTS = {
  'Dashboard': Dashboard,
  'Live Scan': LiveScan,
  'Federated': FederatedLearning,
  'Security': Security,
  'System': System,
}

export default function App() {
  const [tab, setTab] = useState('Live Scan')
  const [health, setHealth] = useState(null)

  useEffect(() => {
    fetch('/health').then(r => r.json()).then(setHealth).catch(() => setHealth({ status: 'offline' }))
    const id = setInterval(() => fetch('/health').then(r => r.json()).then(setHealth).catch(() => {}), 8000)
    return () => clearInterval(id)
  }, [])

  const Comp = COMPONENTS[tab]
  const live = health?.status === 'ok'

  return (
    <>
      <header>
        <div className="header-brand">
          <div className="header-logo">FS</div>
          <div>
            <h1>FedShield Control Center</h1>
            <p>Endpoint · Federated Learning · Live file monitoring — telemetry only, no raw uploads</p>
          </div>
        </div>
        <div className="header-meta">
          <span className="header-badge">{live ? '● Live' : '○ Offline'} · {health?.secure ? 'Secure' : 'Plain'} · {health?.port || 8000}</span>
          <span className={`dot ${live ? '' : 'warn'}`} />
        </div>
      </header>
      <nav>
        {TABS.map(t => (
          <button key={t.id} className={tab === t.id ? 'active' : ''} onClick={() => setTab(t.id)}>
            <span className="ico">{t.ico}</span> {t.label}
          </button>
        ))}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
          <span className="badge neutral">5 tabs · professional view</span>
        </div>
      </nav>
      <main>
        <Comp />
      </main>
    </>
  )
}
