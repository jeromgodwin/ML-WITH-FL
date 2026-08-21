import React, { useEffect, useState } from 'react'

export default function EndpointStatus() {
  const [status, setStatus] = useState(null)
  useEffect(() => {
    fetch('/api/v1/protection/status').then(r => r.json()).then(setStatus).catch(() => {})
  }, [])
  return (
    <div className="card">
      <h3>Endpoint Protection Status — high-level, no raw-file management</h3>
      <div>Status: <span className="badge ok">{status?.protection || 'active'}</span></div>
      <div>Active Model: {status?.active_model || '—'}</div>
      <div>Detections: {status?.detections_count ?? '—'} (telemetry only, no raw files)</div>
      <div>Quarantine: {status?.quarantined_count ?? '—'} files</div>
      <div>Note: Control center is not a file-upload antivirus — endpoint handles files automatically.</div>
    </div>
  )
}
