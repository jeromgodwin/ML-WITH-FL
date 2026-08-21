import React, { useEffect, useState } from 'react'

export default function AlgorithmComparison() {
  const [data, setData] = useState([])
  useEffect(() => {
    fetch('/api/v1/fl/comparison').then(r => r.json()).then(d => setData(Array.isArray(d) ? d : d.rows || [])).catch(() => {})
  }, [])
  const centralized = data.find(d => d.algorithm === 'centralized')
  const fedavg = data.filter(d => d.algorithm === 'fedavg')
  const fedprox = data.filter(d => d.algorithm === 'fedprox')
  const personalized = data.filter(d => d.algorithm === 'personalized')
  return (
    <div className="card">
      <h3>Algorithm Comparison — centralized vs FedAvg / FedProx / personalized</h3>
      <table>
        <thead><tr><th>Algorithm</th><th>F1</th><th>Accuracy</th><th>Rounds</th></tr></thead>
        <tbody>
          {[centralized, ...fedavg, ...fedprox, ...personalized].filter(Boolean).map((c, i) => (
            <tr key={i}><td>{c.algorithm || c.experiment_id}</td><td>{c.f1?.toFixed(3) || '—'}</td><td>{c.accuracy?.toFixed(3) || '—'}</td><td>{c.rounds || '—'}</td></tr>
          ))}
        </tbody>
      </table>
      {data.length === 0 && <div>No comparison data yet</div>}
    </div>
  )
}
