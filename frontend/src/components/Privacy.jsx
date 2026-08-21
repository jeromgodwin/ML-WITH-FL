import React, { useEffect, useState } from 'react'

export default function Privacy() {
  const [data, setData] = useState([])
  useEffect(() => {
    fetch('/api/v1/fl/comparison').then(r => r.json()).then(d => setData(Array.isArray(d) ? d : [])).catch(() => {})
  }, [])
  return (
    <div className="card">
      <h3>Privacy Setting vs Utility {data.length === 0 && '(no DP data)'}</h3>
      <table>
        <thead><tr><th>Experiment</th><th>Privacy</th><th>F1</th><th>Delta F1</th></tr></thead>
        <tbody>
          {data.map(c => (
            <tr key={c.experiment_id}><td>{c.experiment_id}</td><td>{c.privacy_enabled ? `sigma ${c.sigma}` : 'No DP'}</td><td>{c.f1?.toFixed(3) || '—'}</td><td>{c.delta_f1_vs_no_dp?.toFixed(3) || '—'}</td></tr>
          ))}
        </tbody>
      </table>
      <div>Stronger privacy (larger sigma) → lower utility. See Phase 17 analysis.</div>
    </div>
  )
}
