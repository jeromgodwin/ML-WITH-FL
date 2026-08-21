import React, { useEffect, useState } from 'react'

export default function FederatedLearning() {
  const [status, setStatus] = useState(null)
  const [comparison, setComparison] = useState([])

  useEffect(() => {
    fetch('/api/v1/fl/comparison').then(r => r.json()).then(setComparison).catch(() => {})
    fetch('/api/v1/status').then(r => r.json()).then(setStatus).catch(() => {})
  }, [])

  return (
    <div>
      <div className="grid">
        <div className="card"><h3>Algorithm</h3><div>{status?.algorithm || 'fedavg'}</div></div>
        <div className="card"><h3>Rounds</h3><div>{status?.rounds || '—'}</div></div>
        <div className="card"><h3>Current Round</h3><div>{status?.current_round || '—'}</div></div>
        <div className="card"><h3>Clients</h3><div>{status?.clients || '—'}</div></div>
      </div>
      <div className="card">
        <h3>Global & Client F1</h3>
        <table>
          <thead><tr><th>Experiment</th><th>Global Acc</th><th>Global F1</th><th>Mean Client F1</th><th>Worst Client F1</th></tr></thead>
          <tbody>
            {comparison.map(c => (
              <tr key={c.experiment_id}><td>{c.experiment_id}</td><td>{c.accuracy?.toFixed(3) || '—'}</td><td>{c.f1?.toFixed(3) || '—'}</td><td>{c.mean_client_f1?.toFixed(3) || '—'}</td><td>{c.worst_client_f1?.toFixed(3) || '—'}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
