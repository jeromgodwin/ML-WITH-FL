import React, { useEffect, useState } from 'react'

export default function Security() {
  const [models, setModels] = useState([])
  useEffect(() => {
    fetch('/api/v1/models').then(r => r.json()).then(setModels).catch(() => {})
  }, [])
  const suspicious = models.filter(m => m.status === 'REJECTED' || m.validation_notes)
  const rejected = models.filter(m => m.status === 'REJECTED')
  return (
    <div>
      <div className="card"><h3>Suspicious Client Updates — anomaly scores</h3><div>See FL round metrics for anomaly scores</div></div>
      <div className="card">
        <h3>Rejected Candidate Models & Validation Failures</h3>
        <table>
          <thead><tr><th>Version</th><th>Status</th><th>Validation Notes</th></tr></thead>
          <tbody>
            {rejected.map(m => <tr key={m.version}><td>{m.version}</td><td><span className="badge err">{m.status}</span></td><td>{m.validation_notes || '—'}</td></tr>)}
          </tbody>
        </table>
        {rejected.length === 0 && <div>No rejected models</div>}
      </div>
      <div className="card"><h3>All Models Suspicious</h3><pre>{JSON.stringify(suspicious.slice(0, 3), null, 2)}</pre></div>
    </div>
  )
}
