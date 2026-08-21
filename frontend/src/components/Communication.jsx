import React, { useEffect, useState } from 'react'

export default function Communication() {
  const [comparison, setComparison] = useState([])
  useEffect(() => {
    fetch('/api/v1/fl/comparison').then(r => r.json()).then(d => setComparison(Array.isArray(d) ? d : [])).catch(() => {})
  }, [])
  return (
    <div>
      <div className="card">
        <h3>Bytes / Round & Total Communication</h3>
        <table>
          <thead><tr><th>Experiment</th><th>Bytes/Round</th><th>Total</th><th>Clients</th></tr></thead>
          <tbody>
            {comparison.map(c => (
              <tr key={c.experiment_id}><td>{c.experiment_id}</td><td>{c.bytes_per_round || c.total_bytes ? (c.total_bytes / (c.rounds || 1)).toFixed(0) : '—'}</td><td>{c.total_bytes || '—'}</td><td>{c.n_clients || '—'}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="card"><h3>Communication / Performance Tradeoff</h3><div>Higher rounds → higher bytes, diminishing F1 gains. See analysis plots.</div></div>
    </div>
  )
}
