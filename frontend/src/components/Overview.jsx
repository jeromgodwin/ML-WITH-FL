import React, { useEffect, useState } from 'react'
import { api } from '../api'

export default function Overview() {
  const [health, setHealth] = useState(null)
  const [status, setStatus] = useState(null)
  const [protection, setProtection] = useState(null)

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: 'offline' }))
    api.serverStatus().then(setStatus).catch(() => {})
    api.protectionStatus().then(setProtection).catch(() => {})
  }, [])

  return (
    <div className="grid">
      <div className="card">
        <h3>Server Status</h3>
        <pre>{JSON.stringify(health || status, null, 2)}</pre>
      </div>
      <div className="card">
        <h3>Active Global Model</h3>
        <pre>{JSON.stringify(protection?.model || protection, null, 2)}</pre>
        <div>Version: {protection?.active_model || '—'}</div>
      </div>
      <div className="card">
        <h3>Latest FL Round</h3>
        <div>{protection?.latest_round || '—'}</div>
      </div>
      <div className="card">
        <h3>Threats / Detection Summaries</h3>
        <div>{protection?.threats || 'Telemetry disabled or no threats'}</div>
      </div>
    </div>
  )
}
