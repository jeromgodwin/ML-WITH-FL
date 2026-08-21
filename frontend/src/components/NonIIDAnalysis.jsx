import React, { useEffect, useState } from 'react'

export default function NonIIDAnalysis() {
  const [data, setData] = useState([])
  useEffect(() => {
    fetch('/api/v1/fl/comparison').then(r => r.json()).then(d => setData(Array.isArray(d) ? d : [])).catch(() => {})
  }, [])
  const bySeverity = { IID: [], mild: [], moderate: [], severe: [] }
  data.forEach(d => {
    const s = (d.strategy || d.non_iid_severity || 'unknown').toLowerCase()
    if (bySeverity[s] !== undefined) bySeverity[s].push(d)
    else bySeverity[s] = [d]
  })
  return (
    <div className="card">
      <h3>Non-IID Analysis — IID vs mild / moderate / severe</h3>
      <table>
        <thead><tr><th>Severity</th><th>Count</th><th>Avg F1</th><th>Avg Accuracy</th></tr></thead>
        <tbody>
          {Object.entries(bySeverity).map(([k, v]) => (
            <tr key={k}><td>{k}</td><td>{v.length}</td><td>{v.length ? (v.reduce((a, c) => a + (c.f1 || 0), 0) / v.length).toFixed(3) : '—'}</td><td>{v.length ? (v.reduce((a, c) => a + (c.accuracy || 0), 0) / v.length).toFixed(3) : '—'}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
