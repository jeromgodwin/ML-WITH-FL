import React, { useEffect, useState } from 'react'

function Badge({ kind, children }) {
  const cls = kind === 'HIGH' || kind === 'err' ? 'badge err' : kind === 'MEDIUM' || kind === 'warn' ? 'badge warn' : kind === 'LOW' || kind === 'ok' ? 'badge ok' : 'badge neutral'
  return <span className={cls}>{children}</span>
}
function RiskBar({ p, level }) {
  if (p == null) return <span className="muted">—</span>
  const pct = Math.round(p * 100)
  const cls = level === 'HIGH' || pct >= 70 ? 'bar bar-risk-high' : level === 'MEDIUM' || pct >= 30 ? 'bar bar-risk-med' : 'bar bar-risk-low'
  return <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}><div className={cls} style={{ width: 90 }}><i style={{ width: pct + '%' }} /></div><b style={{ fontSize: 12 }}>{pct}%</b></div>
}

export default function LiveScan() {
  const [status, setStatus] = useState(null)
  const [filter, setFilter] = useState('all')
  const [q, setQ] = useState('')
  const [sel, setSel] = useState(null)

  const load = () => fetch('/api/v1/monitor/status').then(r => r.json()).then(setStatus).catch(() => {})
  useEffect(() => { load(); const id = setInterval(load, 2000); return () => clearInterval(id) }, [])

  const allFiles = status?.recent_all_files || []
  const recentEvents = status?.recent_events || []
  const detections = status?.recent_detections || []
  const watched = status?.watched_directories || []

  // unify feed: prefer allFiles, fallback to detections
  let feed = allFiles.length ? allFiles : detections.map(d => ({ ...d, filename: d.filename || d.path?.split(/[\\/]/).pop(), scan_status: 'scanned', is_pe: true, vulnerability: { malware_probability: d.malware_probability, risk_score: d.risk_score, verdict: d.verdict, action: d.action, risk_level: d.risk_level, explanation: d.explanation } }))

  if (filter === 'pe') feed = feed.filter(f => f.is_pe)
  if (filter === 'skipped') feed = feed.filter(f => f.scan_status === 'skipped')
  if (filter === 'threats') feed = feed.filter(f => (f.vulnerability?.verdict === 'HIGH') || (f.verdict === 'HIGH') || f.vulnerability?.risk_level === 'HIGH' || (f.vulnerability?.malware_probability||0) >= 0.7)
  if (q) { const qq = q.toLowerCase(); feed = feed.filter(f => (f.filename||'').toLowerCase().includes(qq) || (f.path||'').toLowerCase().includes(qq) || (f.sha256||'').toLowerCase().includes(qq)) }

  const kpiAll = status?.all_files_seen ?? allFiles.length
  const kpiPe = status?.files_seen ?? recentEvents.length
  const kpiAnalyzed = status?.files_analyzed ?? 0
  const kpiQ = status?.quarantined_count ?? detections.filter(d => d.action === 'QUARANTINE').length

  return (
    <div>
      {/* watched dirs + controls */}
      <div className="card">
        <div className="kicker">Live file monitor · every arrival is visible</div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            {watched.map(p => <span key={p} className="feature-chip">📁 {p}</span>)}
            {!watched.length && <span className="muted">No watched directories</span>}
            <Badge kind={status?.running ? 'ok' : 'warn'}>{status?.running ? '● running' : '○ idle'}</Badge>
            <span className="muted">poll {status?.poll_interval || 1}s · stability 2s · MZ sniff</span>
          </div>
          <div className="hint">Drop any file into a watched dir — PE (.exe/.dll) gets full scan + vulnerability, others show as “skipped / N/A”</div>
        </div>
      </div>

      {/* KPIs */}
      <div className="grid">
        <div className="card kpi"><div className="kpi-label">All arrivals</div><div className="kpi-value">{kpiAll}</div><div className="kpi-sub">Every file that reached a watched dir (PE + non-PE)</div></div>
        <div className="card kpi"><div className="kpi-label">PE candidates</div><div className="kpi-value">{kpiPe}</div><div className="kpi-sub">Extension or MZ signature — sent to malware scan</div></div>
        <div className="card kpi"><div className="kpi-label">Scanned</div><div className="kpi-value">{kpiAnalyzed || feed.filter(f=>f.scan_status==='scanned').length}</div><div className="kpi-sub">Static analysis only — never executed</div></div>
        <div className="card kpi"><div className="kpi-label">Quarantined</div><div className="kpi-value">{kpiQ}</div><div className="kpi-sub">Action=QUARANTINE · {status?.active_model ? status.active_model : 'model mlp-central-v1 active'}</div></div>
      </div>

      <div className="grid-2">
        {/* Feed */}
        <div className="card">
          <h3>File arrival feed <small>· auto-refresh 2s · click row for scan pipeline</small></h3>
          <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap', alignItems: 'center' }}>
            <div className="filters">
              {['all','pe','threats','skipped'].map(k => <button key={k} className={filter===k?'active':''} onClick={()=>setFilter(k)}>{k==='all'?'All':k==='pe'?'PE scanned':k==='threats'?'Threats':'Skipped'}</button>)}
            </div>
            <input className="input" placeholder="Search filename / sha256 / path" value={q} onChange={e=>setQ(e.target.value)} />
            <span className="muted">{feed.length} shown</span>
          </div>

          <div className="table-wrap">
            <table>
              <thead><tr><th>Time</th><th>File</th><th>Type</th><th>Scan</th><th>Malware prob.</th><th>Vulnerability</th><th>Action</th></tr></thead>
              <tbody>
                {feed.map((f, i) => {
                  const v = f.vulnerability || {}
                  const p = v.malware_probability ?? f.malware_probability
                  const verdict = v.verdict || f.verdict
                  const action = v.action || f.action
                  const level = v.risk_level || f.risk_level || (p>=0.7?'HIGH':p>=0.3?'MEDIUM':'LOW')
                  const isPe = f.is_pe ?? (f.detected_by && f.detected_by !== 'none')
                  return (
                    <tr key={(f.sha256||f.path||i)+i} onClick={()=>setSel(f)} className={sel?.path===f.path?'row-selected':''} style={{ cursor: 'pointer' }}>
                      <td style={{ whiteSpace: 'nowrap', fontVariantNumeric: 'tabular-nums' }}>{new Date((f.detected_at||0)*1000).toLocaleTimeString([], { hour12: false }) || '—'} <div className="muted" style={{ fontSize: 11 }}>{f.size != null ? (f.size/1024).toFixed(1)+' KB' : ''}</div></td>
                      <td><b style={{ fontSize: 13 }}>{f.filename || f.path?.split(/[\\/]/).pop()}</b><div className="muted" style={{ fontSize: 11, maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.path}</div><div className="muted" style={{ fontSize: 10, fontFamily: 'monospace' }}>{(f.sha256||'').slice(0,12)}</div></td>
                      <td><Badge kind="neutral">{f.extension || f.file_type || (isPe?'pe':'—')}</Badge><div className="muted" style={{ fontSize: 11 }}>{f.detected_by || (isPe?'pe':'—')}</div></td>
                      <td>{f.scan_status==='scanned'?<Badge kind={verdict==='HIGH'?'err':verdict==='MEDIUM'?'warn':'ok'}>scanned</Badge>:f.scan_status==='skipped'?<Badge kind="neutral">skipped</Badge>:<Badge kind="warn">{f.scan_status||'pending'}</Badge>}</td>
                      <td><RiskBar p={p} level={level} /><div className="muted" style={{ fontSize: 11 }}>{v.risk_score != null ? 'risk '+v.risk_score : ''} {level ? '· '+level : ''}</div></td>
                      <td>
                        {f.scan_status==='skipped' ? <span className="badge neutral">N/A — not PE</span> : verdict ? <Badge kind={verdict}>{verdict}</Badge> : <span className="muted">pending</span>}
                        {v.explanation?.top_features ? <div style={{ marginTop: 4, display: 'flex', gap: 4, flexWrap: 'wrap' }}>{v.explanation.top_features.slice(0,2).map(tf => <span key={tf[0]} className="feature-chip">{tf[0]}</span>)}</div> : null}
                      </td>
                      <td>{action ? <Badge kind={action==='QUARANTINE'?'err':action==='WARN'?'warn':'ok'}>{action}</Badge> : <span className="muted">—</span>}</td>
                    </tr>
                  )
                })}
                {!feed.length && <tr><td colSpan={7}><div className="empty"><b>No arrivals yet</b><div>Copy any file into <code>{watched[0]||'D:/Telegram'}</code> — every arrival appears here. PE files show full scan + vulnerability; images/docs show as “skipped / N/A”.</div></div></td></tr>}
              </tbody>
            </table>
          </div>
          <div className="hint">All arrivals are persisted in <code>data/monitor/status.json</code> and <code>data/monitor/detections.jsonl</code> — no raw file ever leaves the endpoint.</div>
        </div>

        {/* Detail + vulnerability */}
        <div>
          <div className="card">
            <h3>Malware scan process {sel ? `· ${sel.filename}` : ''}</h3>
            {!sel ? <div className="empty"><b>Select a file</b><div>Click any row — see 7-step static pipeline and vulnerability breakdown.</div></div> : (
              <div>
                <div className="timeline" style={{ marginBottom: 12 }}>
                  {(sel.vulnerability?.scan_process || ["stable","sha256","feature_extraction","inference","risk_engine","verdict","quarantine_check"]).map((s, idx) => {
                    const done = sel.scan_status==='scanned'
                    const active = idx===2 && sel.scan_status==='scanned'
                    return <span key={s} className={`step ${done?'done':''} ${active?'active':''}`}><span className="n">{idx+1}</span>{s.replace('_',' ')}</span>
                  })}
                </div>

                <div className={`vuln-card ${sel.vulnerability?.risk_level==='HIGH'||sel.vulnerability?.verdict==='HIGH'?'high':sel.vulnerability?.risk_level==='MEDIUM'?'med':'low'}`} style={{ marginBottom: 12 }}>
                  <div className="kicker">Vulnerability</div>
                  {sel.scan_status==='skipped' ? (
                    <div><b>{sel.vulnerability?.note || 'Non-executable file'}</b><div className="muted">{sel.scan_reason}</div><div className="muted">Attack surface: {sel.vulnerability?.attack_surface||'none'} · No PE static features to assess.</div></div>
                  ) : (
                    <div>
                      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
                        <Badge kind={sel.vulnerability?.verdict || sel.verdict}>{sel.vulnerability?.verdict || sel.verdict || '—'}</Badge>
                        <Badge kind={sel.vulnerability?.risk_level || sel.risk_level}>{sel.vulnerability?.risk_level || sel.risk_level || '—'} · risk {sel.vulnerability?.risk_score ?? sel.risk_score ?? '—'}</Badge>
                        <Badge kind={sel.vulnerability?.action || sel.action}>{sel.vulnerability?.action || sel.action || '—'}</Badge>
                      </div>
                      <div style={{ marginTop: 8 }}><RiskBar p={sel.vulnerability?.malware_probability ?? sel.malware_probability} level={sel.vulnerability?.risk_level} /></div>
                      <div className="muted" style={{ marginTop: 6 }}>Model <b>{sel.vulnerability?.model_version || sel.model_version || '—'}</b> · {sel.vulnerability?.analysis_duration_ms || sel.analysis_duration_ms ? (sel.vulnerability?.analysis_duration_ms || sel.analysis_duration_ms)+' ms' : ''} {sel.vulnerability?.file_type ? '· '+sel.vulnerability.file_type : ''}</div>
                      {sel.vulnerability?.explanation?.top_features && (
                        <div style={{ marginTop: 10 }}>
                          <div className="kicker">Top contributing signals (not proof of maliciousness)</div>
                          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                            {sel.vulnerability.explanation.top_features.map(([name, val]) => <span key={name} className="feature-chip">{name}: {typeof val==='number'?val.toFixed(3):String(val).slice(0,12)}</span>)}
                          </div>
                          <div className="muted" style={{ marginTop: 6, fontSize: 11 }}>{sel.vulnerability.explanation.note}</div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <div style={{ display: 'grid', gap: 6, fontSize: 12 }}>
                  <div><b>Path</b> <span className="muted" style={{ wordBreak: 'break-all' }}>{sel.path}</span></div>
                  <div><b>SHA256</b> <span style={{ fontFamily: 'monospace', fontSize: 11 }}>{sel.sha256}</span></div>
                  <div><b>Size</b> {sel.size} bytes · <b>Ext</b> {sel.extension || sel.file_type} · <b>Detected by</b> {sel.detected_by || '—'} · <b>Detected at</b> {sel.detected_at}</div>
                  {sel.scan_reason && <div className="muted"><b>Reason:</b> {sel.scan_reason}</div>}
                </div>
              </div>
            )}
          </div>

          <div className="card">
            <h3>Quarantine <small>· {status?.quarantined_count ?? 0} files</small></h3>
            <div className="muted" style={{ fontSize: 12 }}>Files with `action=QUARANTINE` are moved to <code>quarantine/</code> with metadata JSON — never executed.</div>
            <div style={{ marginTop: 8 }}>
              {(status?.recent_detections||[]).filter(d=>d.action==='QUARANTINE').slice(0,5).map(d => (
                <div key={d.detection_id||d.sha256} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--border)', fontSize: 12 }}>
                  <span>{d.filename || d.sha256?.slice(0,12)} <span className="muted">· {d.file_type}</span></span><Badge kind="err">QUARANTINE</Badge>
                </div>
              ))}
              {!(status?.recent_detections||[]).some(d=>d.action==='QUARANTINE') && <div className="muted" style={{ padding: '8px 0' }}>No quarantined files yet.</div>}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
