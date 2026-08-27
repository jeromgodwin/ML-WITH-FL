import React, { useEffect, useState } from 'react'

export default function Dashboard() {
  const [health, setHealth] = useState(null)
  const [prot, setProt] = useState(null)
  const [mon, setMon] = useState(null)
  const [fl, setFl] = useState([])
  useEffect(() => {
    fetch('/health').then(r=>r.json()).then(setHealth).catch(()=>{})
    fetch('/api/v1/protection/status').then(r=>r.json()).then(setProt).catch(()=>{})
    fetch('/api/v1/monitor/status').then(r=>r.json()).then(setMon).catch(()=>{})
    fetch('/api/v1/fl/comparison').then(r=>r.json()).then(setFl).catch(()=>{})
  }, [])

  const active = prot?.active_model
  const allFiles = mon?.all_files_seen ?? 0
  const peFiles = mon?.files_seen ?? 0
  const running = mon?.running

  return (
    <div>
      <div className="grid">
        <div className="card kpi">
          <div className="kpi-label">Protection</div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span className={`badge ${prot?.protection==='active'?'ok':'warn'}`}>{prot?.protection || '—'}</span>
            <span className="muted">{active || 'no_active_model'}</span>
          </div>
          <div className="kpi-sub" style={{ marginTop: 6 }}>Model: {prot?.model?.algorithm || 'centralized'} · {active ? 'ready for local inference' : 'train or promote a model'}</div>
          <div className="hint">Static analysis only — files never executed.</div>
        </div>
        <div className="card kpi">
          <div className="kpi-label">Live monitor</div>
          <div className="kpi-value" style={{ fontSize: 18 }}>{running ? '● Running' : '○ Idle'} <span style={{ fontSize: 12, fontWeight: 600, color: '#64748b' }}>{(mon?.watched_directories||[]).length} dirs</span></div>
          <div className="kpi-sub">{(mon?.watched_directories||[]).join(' · ') || '—'}</div>
          <div className="kpi-sub">All arrivals {allFiles} · PE {peFiles} · analyzed {mon?.files_analyzed||0}</div>
        </div>
        <div className="card kpi">
          <div className="kpi-label">Server</div>
          <div className="kpi-value" style={{ fontSize: 16 }}>{health?.host||'127.0.0.1'}:{health?.port||8000}</div>
          <div className="kpi-sub">Secure: {String(health?.secure)} · {health?.status==='ok'?'● Live':'○ Offline'}</div>
          <div className="hint">Control center is not a file-upload antivirus — endpoint handles files automatically.</div>
        </div>
        <div className="card kpi">
          <div className="kpi-label">Quarantine & Threats</div>
          <div className="kpi-value">{mon?.quarantined_count ?? (mon?.recent_detections||[]).filter(d=>d.action==='QUARANTINE').length}</div>
          <div className="kpi-sub">Recent detections {(mon?.recent_detections||[]).length} · last scan {mon?.last_scan_at ? new Date(mon.last_scan_at*1000).toLocaleTimeString() : '—'}</div>
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <h3>How it works <small>· every file is visible</small></h3>
          <div className="timeline" style={{ marginBottom: 10 }}>
            {["arrival","stable 2s","MZ sniff","scan","vulnerability","quarantine"].map((s,i)=><span key={s} className="step done"><span className="n">{i+1}</span>{s}</span>)}
          </div>
          <div className="muted" style={{ fontSize: 12, lineHeight: 1.6 }}>
            Any file dropped into a watched directory appears in <b>Live Scan</b> within 2–3s. PE executables get full static analysis (2381 features → MLP → risk engine → verdict/action + top 3 signals). Non-PE files show as <span className="badge neutral">skipped / N/A</span> — no false vulnerability.
            Open <b>Live Scan</b> → drop a file into <code>D:\Telegram</code> or <code>D:\Downloads</code> to see it instantly.
          </div>
        </div>
        <div className="card">
          <h3>Recent detections <small>· vulnerability preview</small></h3>
          {(mon?.recent_detections||[]).slice(0,5).map(d=>(
            <div key={d.detection_id||d.sha256} style={{ display: 'flex', justifyContent: 'space-between', padding: '7px 0', borderBottom: '1px solid var(--border)' }}>
              <div><b style={{ fontSize: 12 }}>{d.filename || d.sha256?.slice(0,10)}</b> <span className="muted" style={{ fontSize: 11 }}>· {(d.malware_probability*100).toFixed(1)}% · {d.file_type}</span></div>
              <span className={`badge ${d.verdict==='HIGH'?'err':d.verdict==='MEDIUM'?'warn':'ok'}`}>{d.verdict||'—'}</span>
            </div>
          ))}
          {!(mon?.recent_detections||[]).length && <div className="empty"><b>No detections yet</b><div>Run <code>python scripts/run_monitor.py</code> and copy a PE stub to see it.</div></div>}
          <div className="hint">Telemetry only — SHA, probability, risk, verdict, top features. No raw bytes.</div>
        </div>
      </div>

      <div className="card">
        <h3>Federated overview <small>· {fl.length} experiments</small></h3>
        <div className="table-wrap"><table><thead><tr><th>Experiment</th><th>Acc</th><th>F1</th></tr></thead><tbody>{fl.slice(0,6).map(c=> <tr key={c.experiment_id}><td>{c.experiment_id}</td><td>{c.accuracy?.toFixed(3)||'—'}</td><td>{c.f1?.toFixed(3)||'—'}</td></tr>)} {!fl.length && <tr><td colSpan={3} className="muted">No FL experiments yet — run <code>scripts/run_experiment.py --set fl.num_rounds=2</code></td></tr>}</tbody></table></div>
      </div>
    </div>
  )
}
