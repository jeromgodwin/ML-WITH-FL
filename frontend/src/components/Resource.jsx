import React, { useEffect, useState } from 'react'

export default function Resource() {
  const [status, setStatus] = useState(null)
  const [metrics, setMetrics] = useState(null)
  useEffect(() => {
    fetch('/api/v1/resource/status').then(r => r.json()).then(setStatus).catch(() => {})
    fetch('/api/v1/resource/metrics').then(r => r.json()).then(setMetrics).catch(() => {})
  }, [])
  return (
    <div className="grid">
      <div className="card"><h3>Aggregate Resource Metrics</h3><pre>{JSON.stringify(metrics, null, 2)}</pre></div>
      <div className="card"><h3>FL Status</h3><div>{status?.training_status || 'idle'}</div></div>
      <div className="card"><h3>Paused / Deferred Reasons</h3><div>{status?.reason || metrics?.reason || '—'}</div></div>
    </div>
  )
}
