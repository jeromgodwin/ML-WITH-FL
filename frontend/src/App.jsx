import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ParallaxBackground from './components/ParallaxBackground'
import Sidebar from './components/Sidebar'
import BentoCard, { BentoGrid } from './components/BentoCard'
import ForensicPipeline from './components/ForensicPipeline'

// Global helper
const valOrDash = (v) => v == null || v === 0 ? '—' : String(v)

const stateFor = (f) => {
  if (!f) return 'WAITING'
  if (f.scan_status === 'skipped') return f.is_pe === false ? 'BENIGN' : 'QUEUED'
  if (f.vulnerability?.verdict === 'HIGH' || f.verdict === 'HIGH') return 'MALICIOUS'
  if (f.vulnerability?.verdict === 'MEDIUM' || f.verdict === 'MEDIUM') return 'SUSPICIOUS'
  if (f.vulnerability?.verdict === 'ERROR' || f.verdict === 'ERROR' || f.scan_status === 'error') return 'FAILED'
  if (f.vulnerability?.verdict === 'LOW' || f.verdict === 'LOW') return 'BENIGN'
  if (f.scan_status === 'quarantined' || f.action === 'QUARANTINE') return 'QUARANTINED'
  return 'SCANNING'
}

function OverviewView({ stats, events, monitorFiles, setSelectedFile, setActive }) {
  return (
    <div className="space-y-6 pb-20">
      <div className="flex items-center gap-2 mb-4">
        <h2 className="text-[11px] font-bold tracking-[0.14em] text-zinc-500 uppercase">Overview</h2>
        <span className="h-px flex-1 bg-white/[0.06]" />
        <span className="text-[11px] text-zinc-500">Workstation</span>
      </div>

      {/* TOP STATISTICS */}
      <BentoGrid className="grid-cols-2 md:grid-cols-4 auto-rows-[120px]">
        {[
          { label: 'FILES DETECTED', v: stats.detected, glow: 'cyan' },
          { label: 'FILES SCANNED', v: stats.scanned, glow: 'cyan' },
          { label: 'THREATS DETECTED', v: stats.threats, glow: 'red' },
          { label: 'QUARANTINED', v: stats.quarantined, glow: 'orange' },
        ].map(c => (
          <BentoCard key={c.label} eyebrow={c.label} title={valOrDash(c.v)} glow={c.glow} className="justify-center text-center">
            <div className="text-3xl font-display font-bold mt-2 tracking-tight text-white">{valOrDash(c.v)}</div>
          </BentoCard>
        ))}
      </BentoGrid>

      <div className="grid grid-cols-1 md:grid-cols-12 gap-5 mt-5">
        <BentoCard span="col-span-1 md:col-span-6" eyebrow="HISTORY" title="RECENT FILES" glow="neutral">
          <div className="recessed-display overflow-x-auto mt-4 max-h-[300px] overflow-y-auto">
            <table className="w-full min-w-[500px] text-left text-xs font-mono">
              <thead className="bg-base-900 border-b border-black/80 text-zinc-500">
                <tr>
                  <th className="p-3 font-bold uppercase tracking-widest">File</th>
                  <th className="p-3 font-bold uppercase tracking-widest">Status</th>
                  <th className="p-3 font-bold uppercase tracking-widest text-right">Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {monitorFiles.slice(0, 8).map((f, i) => {
                  const st = stateFor(f)
                  return (
                    <tr key={i} className="hover:bg-white/[0.02] transition-colors cursor-pointer" onClick={() => { setSelectedFile(f); setActive('monitor'); }}>
                      <td className="p-3 font-bold text-white max-w-[200px] truncate">{f.filename}</td>
                      <td className="p-3">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${st === 'MALICIOUS' ? 'bg-red-500/20 text-red-400' : st === 'SUSPICIOUS' ? 'bg-orange-500/20 text-orange-400' : st === 'BENIGN' ? 'bg-green-500/20 text-green-400' : st === 'SCANNING' ? 'bg-cyan-500/20 text-cyan-400 animate-pulse' : 'bg-zinc-700 text-zinc-400'}`}>
                          {st}
                        </span>
                      </td>
                      <td className="p-3 text-right text-zinc-500">{new Date((f.timestamp || Date.now() / 1000) * 1000).toLocaleTimeString()}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </BentoCard>

        <BentoCard span="col-span-1 md:col-span-6" eyebrow="SYSTEM ACTIVITY" title="REAL-TIME EVENTS" glow="neutral">
          <div className="recessed-display p-4 max-h-[300px] overflow-y-auto mt-2">
            <div className="space-y-3">
              {events.length === 0 ? <div className="text-zinc-500 font-mono text-xs">Waiting for events...</div> :
                events.map((e, idx) => (
                  <div key={idx} className="flex gap-4 text-xs font-mono">
                    <span className="text-zinc-500 shrink-0">{new Date(e.time * 1000).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
                    <span className={`${e.type === 'alert' ? 'text-console-red' : e.type === 'detected' ? 'text-console-blue' : e.type === 'quarantine' ? 'text-console-orange' : 'text-zinc-300'}`}>{e.msg}</span>
                  </div>
                ))
              }
            </div>
          </div>
        </BentoCard>
      </div>
    </div>
  )
}

function LiveMonitorView({ monitorFiles, selectedFile, setSelectedFile, events }) {
  const active = selectedFile

  return (
    <div className="space-y-6 pb-20">
      <div className="flex items-center gap-2 mb-4">
        <h2 className="text-[11px] font-bold tracking-[0.14em] text-zinc-500 uppercase">Live Monitor</h2>
        <span className="h-px flex-1 bg-white/[0.06]" />
        <span className="text-[11px] text-zinc-500">Workstation</span>
      </div>

      <BentoGrid className="grid-cols-1 md:grid-cols-12 auto-rows-auto md:auto-rows-[300px]">
        {/* File Tray */}
        <BentoCard span="col-span-1 md:col-span-5" eyebrow="LIVE FILE TRAY" title="MONITORING" glow="cyan" recessed>
          <div className="space-y-3 max-h-[300px] md:max-h-full overflow-y-auto pr-2 mt-2">
            {monitorFiles.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-zinc-500 font-mono text-xs opacity-50 pt-10">
                <div className="text-2xl mb-2">📥</div>
                WAITING FOR FILES...
              </div>
            ) : monitorFiles.map(f => {
              const st = stateFor(f)
              const isSel = active?.path === f.path || active?.sha256 === f.sha256
              return (
                <button key={f.path || f.sha256} onClick={() => setSelectedFile(f)}
                  className={`w-full text-left p-3 rounded-lg border flex gap-3 items-center transition-all shadow-btn-raised
                      ${isSel ? 'bg-base-700 border-console-cyan/50 shadow-btn-pressed' : 'bg-base-800 border-black/80 hover:bg-base-700'}
                    `}
                >
                  <div className="w-10 h-10 rounded bg-[#060809] shadow-panel-recessed flex items-center justify-center border border-black/80 text-xl shrink-0">📄</div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-semibold text-zinc-200 truncate">{f.filename}</div>
                    <div className="text-xs text-zinc-500 font-mono mt-0.5">{f.size ? (f.size / 1024).toFixed(1) + ' KB' : ''}</div>
                  </div>
                  <div className="text-right shrink-0">
                    <div className={`text-[10px] font-bold px-2 py-0.5 rounded
                        ${st === 'MALICIOUS' ? 'bg-red-500/20 text-red-400' : st === 'SUSPICIOUS' ? 'bg-orange-500/20 text-orange-400' : st === 'BENIGN' ? 'bg-green-500/20 text-green-400' : st === 'SCANNING' ? 'bg-cyan-500/20 text-cyan-400 animate-pulse' : 'bg-zinc-700 text-zinc-400'}
                      `}>{st}</div>
                  </div>
                </button>
              )
            })}
          </div>
        </BentoCard>

        {/* Current Scan */}
        <BentoCard span="col-span-1 md:col-span-7" eyebrow="CURRENT SCAN" title="MALWARE ANALYSIS UNIT" glow="cyan">
          {!active ? (
            <div className="h-full flex flex-col items-center justify-center text-zinc-500 font-mono text-sm opacity-50">
              SCANNER READY
              <div className="text-xs mt-1">No active analysis.</div>
            </div>
          ) : (
            <div className="flex flex-col h-full">
              <div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase font-mono mb-2">FILE</div>
              <div className="text-xl font-mono font-bold text-white tracking-wide truncate">{active.filename}</div>
              <div className="text-xs font-mono text-zinc-500 mt-2 truncate max-w-[80%]">{active.path || '—'} {active.size ? `- ${active.size} bytes` : ''}</div>
              <div className="text-xs font-mono text-zinc-500 mt-1 truncate">SHA {active.sha256 || '—'}</div>
              
              <div className="mt-8">
                <div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase font-mono mb-2 flex justify-between">
                  <span>STATUS</span>
                  <div className={`flex items-center gap-2 ${stateFor(active)==='SCANNING'?'text-console-cyan':stateFor(active)==='MALICIOUS'?'text-console-red':'text-console-green'}`}>
                    <div className={`led ${stateFor(active) === 'SCANNING' ? 'led-cyan animate-pulse' : stateFor(active) === 'MALICIOUS' ? 'led-red' : 'led-green'}`} />
                    {stateFor(active)}
                  </div>
                </div>
                <div className="h-2 w-full bg-[#060809] shadow-inner rounded-full overflow-hidden border border-black/80">
                   <div className="h-full bg-console-cyan" style={{width: stateFor(active)==='SCANNING' ? '40%' : '100%'}} />
                </div>
              </div>

              <div className="flex gap-10 mt-6 pt-4 border-t border-white/5">
                <div>
                  <div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase font-mono mb-1">STAGE</div>
                  <div className="text-sm font-mono text-white">
                    {active.vulnerability?.analysis_duration_ms 
                      ? (active.vulnerability.analysis_duration_ms/1000).toFixed(2)+'s' 
                      : active.analysis_duration_ms 
                        ? (active.analysis_duration_ms/1000).toFixed(2)+'s' 
                        : (stateFor(active) === 'SCANNING' ? 'PROCESSING...' : '—')}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase font-mono mb-1">MODEL</div>
                  <div className="text-sm font-mono text-white">{active.vulnerability?.model_version || active.model_version || '—'}</div>
                </div>
              </div>
            </div>
          )}
        </BentoCard>
      </BentoGrid>

      {/* PIPELINE STAGES */}
      <div className="mt-8">
          <ForensicPipeline activeFile={active} events={events} />
      </div>
    </div>
  )
}

function FileAnalysisView({ selectedFile, isQuarantining, handleQuarantine }) {
  const active = selectedFile

  return (
    <div className="space-y-6 pb-20">
      <div className="flex items-center gap-2 mb-4">
        <h2 className="text-[11px] font-bold tracking-[0.14em] text-zinc-500 uppercase">File Analysis</h2>
        <span className="h-px flex-1 bg-white/[0.06]" />
        <span className="text-[11px] text-zinc-500">Workstation</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-12 gap-5 mt-5">
        <BentoCard span="col-span-1 md:col-span-4" eyebrow="SECURITY RESULT" title="MALWARE ANALYSIS" glow="red">
          {!active ? <div className="text-zinc-600 font-mono text-sm mt-4">NO FILE SELECTED</div> : (
            <div className="mt-4 space-y-6">
              <div>
                <div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase font-mono mb-1">PREDICTION</div>
                <div className={`text-2xl font-display font-bold flex items-center gap-2 
                  ${stateFor(active) === 'MALICIOUS' ? 'text-console-red' : stateFor(active) === 'SUSPICIOUS' ? 'text-console-orange' : 'text-console-green'}
                `}>
                  <div className={`led ${stateFor(active) === 'MALICIOUS' ? 'led-red' : stateFor(active) === 'SUSPICIOUS' ? 'led-orange' : 'led-green'}`} />
                  {stateFor(active)}
                </div>
              </div>
              
              <div className="flex gap-10">
                <div>
                  <div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase font-mono mb-1">CONFIDENCE</div>
                  <div className="text-xl font-mono text-white">
                    {active.vulnerability?.malware_probability != null ? (active.vulnerability.malware_probability * 100).toFixed(1) + '%' : '—'}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase font-mono mb-1">MODEL</div>
                  <div className="text-xl font-mono text-white">{active.vulnerability?.model_version || active.model_version || 'XGBoost'}</div>
                </div>
              </div>
              
              {/* Risk Meter */}
              <div className="recessed-display p-5 text-center mt-4">
                <div className="text-4xl font-display font-bold text-white">{active.vulnerability?.risk_score ?? '—'} <span className="text-lg text-zinc-500">/ 100</span></div>
                <div className={`text-sm font-bold mt-2 tracking-widest uppercase ${active.vulnerability?.risk_level === 'HIGH' ? 'text-console-red' : active.vulnerability?.risk_level === 'MEDIUM' ? 'text-console-orange' : 'text-console-green'}`}>
                  {active.vulnerability?.risk_level || 'UNKNOWN'} RISK
                </div>
                <div className="w-full h-2 bg-black rounded-full mt-4 overflow-hidden shadow-inner border border-white/10">
                  <div className="h-full bg-gradient-to-r from-console-green via-console-orange to-console-red" style={{ width: `${active.vulnerability?.risk_score || 0}%` }} />
                </div>
              </div>
              
              <button
                onClick={handleQuarantine}
                disabled={isQuarantining || stateFor(active) === 'QUARANTINED'}
                className="btn-tactile w-full py-4 mt-6 text-sm font-bold tracking-widest uppercase flex justify-center items-center gap-2"
              >
                {stateFor(active) === 'QUARANTINED' ? '✓ QUARANTINED' : isQuarantining ? 'PROCESSING...' : 'QUARANTINE FILE'}
              </button>
            </div>
          )}
        </BentoCard>

        <BentoCard span="col-span-1 md:col-span-8" eyebrow="SECURITY FINDINGS" title="VULNERABILITY ASSESSMENT" glow="neutral">
          {!active ? <div className="text-zinc-600 font-mono text-sm mt-4">NO FILE SELECTED</div> : (
            <div className="mt-4 flex flex-col gap-5">
              
              <div className="recessed-display p-6">
                <div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase font-mono mb-2">FILE DETAILS</div>
                <div className="grid grid-cols-2 gap-4 font-mono text-xs">
                  <div><span className="text-zinc-500">Name:</span> <span className="text-white">{active.filename || '—'}</span></div>
                  <div><span className="text-zinc-500">Size:</span> <span className="text-white">{active.size ? active.size + ' bytes' : '—'}</span></div>
                  <div className="col-span-2 truncate"><span className="text-zinc-500">Path:</span> <span className="text-white">{active.path || '—'}</span></div>
                  <div className="col-span-2 truncate"><span className="text-zinc-500">SHA-256:</span> <span className="text-white">{active.sha256 || '—'}</span></div>
                </div>
              </div>

              <div className="recessed-display p-6 flex-1">
                {active.vulnerability?.note ? (
                  <div className="space-y-6">
                    <div className="flex gap-5 border-b border-white/5 pb-6">
                      <div className={`px-3 py-1 h-fit text-xs font-bold rounded ${active.vulnerability.risk_level === 'HIGH' ? 'bg-red-500/20 text-console-red border border-console-red/30' : active.vulnerability.risk_level === 'MEDIUM' ? 'bg-orange-500/20 text-console-orange border border-console-orange/30' : 'bg-green-500/20 text-console-green border border-console-green/30'}`}>
                        {active.vulnerability.risk_level}
                      </div>
                      <div>
                        <div className="text-base font-bold text-white mb-2">Suspicious PE Characteristics</div>
                        <div className="text-sm text-zinc-400 leading-relaxed max-w-2xl">{active.vulnerability.note}</div>
                        <div className="text-[10px] font-mono text-zinc-500 mt-3 flex items-center gap-2"><div className="led led-red"/> STATUS: CONFIRMED FINDING</div>
                      </div>
                    </div>
                    
                    {/* Explainability / Top Features */}
                    {active.vulnerability?.explanation?.top_features && (
                      <div className="pt-2">
                        <div className="text-xs font-bold text-zinc-300 mb-4 tracking-widest uppercase">Feature Contribution</div>
                        <div className="space-y-3 max-w-xl">
                          {active.vulnerability.explanation.top_features.map((item, idx) => {
                            let name = Array.isArray(item) ? item[0] : (item.feature || `feature_${idx}`);
                            let val = Array.isArray(item) ? item[1] : (item.importance || item.value);
                            return (
                              <div key={idx} className="flex justify-between items-center gap-4">
                                <span className="font-mono text-[11px] text-zinc-400 w-1/3 truncate" title={name}>{name}</span>
                                <div className="flex-1 h-2 bg-black rounded-full overflow-hidden shadow-inner border border-white/5">
                                  <div className="h-full bg-console-blue" style={{ width: `${Math.min(Math.abs(val) * 100, 100)}%` }} />
                                </div>
                                <span className="font-mono text-[11px] text-zinc-500 w-12 text-right">{(Math.abs(val)*100).toFixed(1)}</span>
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-zinc-500 font-mono text-sm h-full flex items-center justify-center">No confirmed vulnerabilities or anomalies detected.</div>
                )}
              </div>
            </div>
          )}
        </BentoCard>
      </div>
    </div>
  )
}

function ThreatsView({ monitorFiles }) {
  return (
    <div className="space-y-6 pb-20">
      <div className="flex items-center gap-2 mb-4">
        <h2 className="text-[11px] font-bold tracking-[0.14em] text-zinc-500 uppercase">Threats</h2>
        <span className="h-px flex-1 bg-white/[0.06]" />
      </div>
      <BentoCard eyebrow="THREAT INTELLIGENCE" title="DETECTED THREATS" glow="red">
        <div className="recessed-display overflow-x-auto mt-4 max-h-[600px] overflow-y-auto">
          <table className="w-full min-w-[600px] text-left text-xs font-mono">
            <thead className="bg-base-900 border-b border-black/80 text-zinc-500">
              <tr>
                <th className="p-3 font-bold uppercase tracking-widest">File</th>
                <th className="p-3 font-bold uppercase tracking-widest">Status</th>
                <th className="p-3 font-bold uppercase tracking-widest">Risk</th>
                <th className="p-3 font-bold uppercase tracking-widest">Confidence</th>
                <th className="p-3 font-bold uppercase tracking-widest text-right">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {monitorFiles.filter(f => stateFor(f) === 'MALICIOUS' || stateFor(f) === 'SUSPICIOUS' || stateFor(f) === 'QUARANTINED').map((f, i) => {
                const st = stateFor(f)
                return (
                  <tr key={i} className="hover:bg-white/[0.02] transition-colors">
                    <td className="p-3 font-bold text-white max-w-[200px] truncate">{f.filename}</td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${st === 'MALICIOUS' ? 'bg-red-500/20 text-red-400' : st === 'SUSPICIOUS' ? 'bg-orange-500/20 text-orange-400' : st === 'QUARANTINED' ? 'bg-zinc-700 text-zinc-400' : 'bg-green-500/20 text-green-400'}`}>
                        {st}
                      </span>
                    </td>
                    <td className="p-3 text-zinc-400">{f.vulnerability?.risk_score ?? '—'}</td>
                    <td className="p-3 text-zinc-400">{f.vulnerability?.malware_probability != null ? (f.vulnerability.malware_probability * 100).toFixed(1) + '%' : '—'}</td>
                    <td className="p-3 text-right text-zinc-500">{new Date((f.timestamp || Date.now() / 1000) * 1000).toLocaleTimeString()}</td>
                  </tr>
                )
              })}
              {monitorFiles.filter(f => stateFor(f) === 'MALICIOUS' || stateFor(f) === 'SUSPICIOUS').length === 0 && (
                <tr><td colSpan="5" className="p-4 text-center text-zinc-600">✓ NO THREATS DETECTED</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </BentoCard>
    </div>
  )
}

function FederatedView({ comp }) {
  return (
    <div className="space-y-6 pb-20">
      <div className="flex items-center gap-2 mb-4">
        <h2 className="text-[11px] font-bold tracking-[0.14em] text-zinc-500 uppercase">Federated Learning</h2>
        <span className="h-px flex-1 bg-white/[0.06]" />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-12 gap-5">
        <BentoCard span="col-span-1 md:col-span-8" eyebrow="FEDERATED LEARNING" title="GLOBAL MODEL STATUS" glow="cyan">
          <div className="mt-4 grid grid-cols-2 gap-6">
            <div className="recessed-display p-6 flex flex-col justify-center items-center">
              <div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase font-mono mb-2">ACCURACY</div>
              <div className="text-5xl font-display font-bold text-console-cyan">{comp?.accuracy ? (comp.accuracy*100).toFixed(1)+'%' : '—'}</div>
            </div>
            <div className="recessed-display p-6 flex flex-col justify-center items-center">
              <div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase font-mono mb-2">F1 SCORE</div>
              <div className="text-5xl font-display font-bold text-console-blue">{comp?.f1 ? (comp.f1*100).toFixed(1)+'%' : '—'}</div>
            </div>
          </div>
          <div className="mt-6 flex justify-between font-mono text-sm border-t border-white/5 pt-6">
            <div><span className="text-zinc-500">MODEL VERSION:</span> <span className="text-white">{comp?.experiment_id?.slice(0,12) || '—'}</span></div>
            <div><span className="text-zinc-500">ALGORITHM:</span> <span className="text-white">FedAvg</span></div>
            <div><span className="text-zinc-500">CLIENTS:</span> <span className="text-white">{comp ? '10' : '—'}</span></div>
          </div>
        </BentoCard>
      </div>
    </div>
  )
}


export default function App() {
  const [active, setActive] = useState('monitor')
  const [system, setSystem] = useState({ online: true, backend: 'CONNECTING' })

  // Shared Data State
  const [stats, setStats] = useState({ detected: null, scanned: null, threats: null, quarantined: null })
  const [monitorFiles, setMonitorFiles] = useState([])
  const [selectedFile, setSelectedFile] = useState(null)
  const [events, setEvents] = useState([])
  const [comp, setComp] = useState(null)
  const [isQuarantining, setIsQuarantining] = useState(false)

  // System status
  useEffect(() => {
    const check = () => fetch('/health').then(r=>r.json()).then(()=> setSystem(s=> ({...s, online:true, backend:'CONNECTED'}))).catch(()=> setSystem(s=> ({...s, online:false, backend:'DISCONNECTED'})))
    check(); const id=setInterval(check, 5000); return()=>clearInterval(id)
  }, [])

  // Live polling
  useEffect(() => {
    const load = () => {
      Promise.all([
        fetch('/api/v1/monitor/status').then(r=>r.json()).catch(()=>null),
        fetch('/api/v1/monitor/files?limit=15').then(r=>r.json()).catch(()=>[]),
        fetch('/api/v1/fl/comparison').then(r=>r.json()).catch(()=>[])
      ]).then(([m, fList, cList]) => {
        if(m) {
          const det=(m.recent_detections||[]).filter(d=>d.verdict==='HIGH'||d.verdict==='MEDIUM').length
          const q=(m.recent_detections||[]).filter(d=>d.action==='QUARANTINE').length
          setStats({ detected: m.all_files_seen, scanned: m.files_analyzed, threats: det, quarantined: q })
        }
        
        const arr = Array.isArray(fList) ? fList : []
        setMonitorFiles(arr)
        
        // Auto-select latest active scan if none selected
        if(arr.length && !selectedFile) setSelectedFile(arr[0])
        else if (selectedFile) {
          const fresh = arr.find(x => x.sha256 === selectedFile.sha256 || x.path === selectedFile.path)
          if(fresh) setSelectedFile(fresh)
        }
        
        // Build timeline
        const evts = []
        arr.forEach(f => {
          const tSec = typeof f.timestamp === 'string' ? (new Date(f.timestamp).getTime() / 1000) : (f.timestamp || (Date.now()/1000));
          if (f.scan_status === 'scanned') evts.push({ time: tSec, msg: `Scan complete: ${f.filename}`, type: 'info' })
          if (f.action === 'QUARANTINE') evts.push({ time: tSec, msg: `File quarantined: ${f.filename}`, type: 'quarantine' })
          if (f.verdict === 'HIGH') evts.push({ time: tSec, msg: `Malicious classification: ${f.filename}`, type: 'alert' })
          if (f.scan_status === 'scanning') evts.push({ time: tSec, msg: `Scan started: ${f.filename}`, type: 'scanning' })
          evts.push({ time: tSec - 1, msg: `New file detected: ${f.filename}`, type: 'detected' })
        })
        setEvents(evts.sort((a,b)=> b.time - a.time).slice(0, 10))
        if(Array.isArray(cList) && cList.length) setComp(cList[0])
      })
    }
    load(); const id = setInterval(load, 1500); return () => clearInterval(id)
  }, [selectedFile])

  const handleQuarantine = () => {
    if(!selectedFile) return
    if(confirm(`Are you sure you want to quarantine ${selectedFile.filename}?`)) {
      setIsQuarantining(true)
      setTimeout(() => {
        setIsQuarantining(false)
        alert('File Quarantined (Demo/UI)')
      }, 1000)
    }
  }

  // Render view router
  const renderView = () => {
    switch (active) {
      case 'overview':
        return <OverviewView stats={stats} events={events} monitorFiles={monitorFiles} setSelectedFile={setSelectedFile} setActive={setActive} />
      case 'monitor':
        return <LiveMonitorView monitorFiles={monitorFiles} selectedFile={selectedFile} setSelectedFile={setSelectedFile} events={events} />
      case 'analysis':
        return <FileAnalysisView selectedFile={selectedFile} isQuarantining={isQuarantining} handleQuarantine={handleQuarantine} />
      case 'threats':
        return <ThreatsView monitorFiles={monitorFiles} />
      case 'federated':
        return <FederatedView comp={comp} />
      default:
        return <div className="recessed-display p-10 text-center font-mono text-zinc-500">Module '{active}' is under construction.</div>
    }
  }

  return (
    <div className="min-h-screen flex text-zinc-300 selection:bg-console-blue/30 selection:text-white font-sans">
      <ParallaxBackground />
      <Sidebar active={active} onSelect={setActive} system={system} />
      
      <div className="flex-1 flex flex-col h-screen overflow-hidden relative z-10">
        <header className="h-[60px] border-b border-black/80 bg-base-950/80 backdrop-blur shadow-[0_2px_10px_rgba(0,0,0,0.5)] flex items-center justify-between px-6 z-20 shrink-0">
          <div className="flex items-center gap-4">
            <h2 className="font-display text-sm font-bold tracking-[0.15em] text-zinc-400 uppercase">
              FEDSHIELD
            </h2>
            <div className="h-4 w-px bg-white/10" />
            <h2 className="font-display text-sm font-bold text-zinc-100 uppercase">
              {active === 'monitor' ? 'Live Monitor' : active === 'analysis' ? 'File Analysis' : active}
            </h2>
            <div className="h-4 w-px bg-white/10 ml-4" />
            <div className="flex items-center gap-3 text-[10px] font-mono font-bold text-zinc-400 bg-white/[0.03] px-3 py-1.5 rounded-full border border-white/5">
              <div className="led led-green" /> SYSTEM ONLINE
            </div>
            <div className="flex items-center gap-3 text-[10px] font-mono font-bold text-zinc-400 bg-white/[0.03] px-3 py-1.5 rounded-full border border-white/5">
              <div className="led led-blue animate-pulse" /> MONITORING ACTIVE
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-[10px] font-mono font-bold text-zinc-500 mr-4">
              <div className="led led-cyan" /> CONNECTED
            </div>
            <div className="w-7 h-7 rounded-full bg-base-800 shadow-panel-raised flex items-center justify-center border border-black/50 text-[10px] font-bold text-zinc-400">👤</div>
            <div className="w-7 h-7 rounded-full bg-console-blue/20 text-console-blue flex items-center justify-center border border-console-blue/30 text-[10px] font-bold">AD</div>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto p-6 scroll-smooth">
          <div className="max-w-[1400px] mx-auto">
            <AnimatePresence mode="wait">
              <motion.div key={active} initial={{ opacity:0, filter:'blur(4px)' }} animate={{ opacity:1, filter:'blur(0px)' }} exit={{ opacity:0, filter:'blur(4px)' }} transition={{ duration:0.2 }}>
                {renderView()}
              </motion.div>
            </AnimatePresence>
          </div>
        </main>
      </div>
    </div>
  )
}
