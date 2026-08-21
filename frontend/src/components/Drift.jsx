import React, { useEffect, useState } from 'react'

export default function Drift() {
  const [status, setStatus] = useState(null)
  const [last, setLast] = useState(null)
  useEffect(() => {
    fetch('/api/v1/drift/status').then(r => r.json()).then(setStatus).catch(() => {})
    fetch('/api/v1/drift/last_event').then(r => r.ok ? r.json() : null).then(setLast).catch(() => {})
  }, [])
  return (
    <div className="grid">
      <div className="card"><h3>Drift Status</h3><div className="badge ok">{status?.drift_status || 'NO_DRIFT'}</div></div>
      <div className="card"><h3>Drift Score</h3><div>{status?.drift_score ?? last?.score ?? '—'}</div></div>
      <div className="card"><h3>Last Event</h3><pre>{JSON.stringify(last, null, 2)}</pre></div>
      <div className="card"><h3>Retraining & Active Model</h3><div>Model: {status?.model_version || '—'}</div></div>
    </div>
  )
}
