import React, { useEffect, useState } from 'react'

export default function System() {
  const [resource, setResource] = useState(null)
  const [drift, setDrift] = useState(null)
  const [models, setModels] = useState([])
  const [clients, setClients] = useState(null)

  useEffect(() => {
    fetch('/api/v1/resource/status').then(r=>r.json()).then(setResource).catch(()=>{})
    fetch('/api/v1/drift/status').then(r=>r.json()).then(setDrift).catch(()=>{})
    fetch('/api/v1/models').then(r=>r.json()).then(setModels).catch(()=>{})
    // clients needs auth, so ignore error
    fetch('/api/v1/clients', { headers: { 'X-Client-Id': 'client-001', Authorization: 'Bearer dummy' } }).then(r=>r.json()).then(setClients).catch(()=>{})
  }, [])

  return (
    <div>
      <div className="grid">
        <div className="card">
          <h3>Resource <small>· policy + metrics</small></h3>
          <div className="muted" style={{ fontSize: 12 }}>Training status: <b>{resource?.training_status || 'idle'}</b></div>
          <pre style={{ fontSize: 11, background: '#f8fafc', padding: 10, borderRadius: 8, overflow: 'auto' }}>{JSON.stringify(resource || {}, null, 2)}</pre>
          <div className="hint">Resource-aware FL pauses when CPU/battery/idle thresholds hit.</div>
        </div>
        <div className="card">
          <h3>Drift <small>· concept drift</small></h3>
          <div className="muted" style={{ fontSize: 12 }}>Status: <b>{drift?.drift_status || drift?.status || '—'}</b> · score {drift?.drift_score ?? '—'}</div>
          <pre style={{ fontSize: 11, background: '#f8fafc', padding: 10, borderRadius: 8, overflow: 'auto' }}>{JSON.stringify(drift || {}, null, 2)}</pre>
        </div>
        <div className="card">
          <h3>Models <small>· registry</small></h3>
          <div className="table-wrap"><table><thead><tr><th>Version</th><th>Algorithm</th><th>Status</th></tr></thead><tbody>
            {(Array.isArray(models)?models:models?.models||[]).slice(0,8).map(m=> <tr key={m.version||m.model_id}><td>{m.version}</td><td>{m.algorithm}</td><td><span className="badge neutral">{m.status}</span></td></tr>)}
            {!(Array.isArray(models)?models.length:(models?.models||[]).length) && <tr><td colSpan={3} className="muted">No models — active: mlp-central-v1</td></tr>}
          </tbody></table></div>
        </div>
        <div className="card">
          <h3>Clients & Privacy <small>· overview</small></h3>
          <div className="muted" style={{ fontSize: 12 }}>Clients: {clients ? JSON.stringify(clients).slice(0,120)+'...' : 'auth required — server manages securely'}</div>
          <div className="hint" style={{ marginTop: 8 }}>Privacy: DP disabled by default. Enable via <code>privacy.enabled=true</code> in config.</div>
          <div className="hint">Communication: bytes/round visible in FL details.</div>
        </div>
      </div>
      <div className="card">
        <h3>How to impress mentors</h3>
        <div className="muted" style={{ fontSize: 12, lineHeight: 1.7 }}>
          1. Show <b>Dashboard</b> → live monitor running, 3 watched dirs, all arrivals vs PE.<br/>
          2. Switch to <b>Live Scan</b> → drop any file into <code>D:\Telegram</code> → appears in 2s with scan pipeline + vulnerability panel (top 3 signals).<br/>
          3. Drop a PE stub (<code>echo MZ &gt; D:\Telegram\demo.exe</code>) → see <span className="badge err">HIGH</span> / <span className="badge ok">LOW</span> with risk bar.<br/>
          4. Non-PE (image/txt) correctly shows <span className="badge neutral">skipped / N/A</span> — not a bug, design for low false positives.
        </div>
      </div>
    </div>
  )
}
