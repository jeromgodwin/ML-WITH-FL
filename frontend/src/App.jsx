import React, { useState } from 'react'
import Overview from './components/Overview'
import Clients from './components/Clients'
import FederatedLearning from './components/FederatedLearning'
import AlgorithmComparison from './components/AlgorithmComparison'
import NonIIDAnalysis from './components/NonIIDAnalysis'
import Resource from './components/Resource'
import Drift from './components/Drift'
import Security from './components/Security'
import Communication from './components/Communication'
import Privacy from './components/Privacy'
import EndpointStatus from './components/EndpointStatus'

const TABS = [
  'Overview', 'Clients', 'Federated Learning', 'Algorithm Comparison', 'Non-IID Analysis',
  'Resource', 'Drift', 'Security', 'Communication', 'Privacy', 'Endpoint Status'
]

const COMPONENTS = {
  'Overview': Overview,
  'Clients': Clients,
  'Federated Learning': FederatedLearning,
  'Algorithm Comparison': AlgorithmComparison,
  'Non-IID Analysis': NonIIDAnalysis,
  'Resource': Resource,
  'Drift': Drift,
  'Security': Security,
  'Communication': Communication,
  'Privacy': Privacy,
  'Endpoint Status': EndpointStatus,
}

export default function App() {
  const [tab, setTab] = useState('Overview')
  const Comp = COMPONENTS[tab]
  return (
    <>
      <header><h1>FedShield Control Center</h1><div>Central server-side — endpoint handles files automatically, no file-upload antivirus</div></header>
      <nav>
        {TABS.map(t => (
          <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>{t}</button>
        ))}
      </nav>
      <main>
        <Comp />
      </main>
    </>
  )
}
