import React, { useState, useEffect, useRef } from 'react';

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

const checkIsComplete = (f) => {
    if (!f) return false;
    const st = stateFor(f);
    return f.vulnerability?.verdict || f.verdict || ['BENIGN', 'MALICIOUS', 'SUSPICIOUS', 'FAILED', 'QUARANTINED'].includes(st);
}

const PHASES = [
  { id: 'detect', label: 'DETECT' },
  { id: 'identify', label: 'IDENTIFY' },
  { id: 'hash', label: 'HASH' },
  { id: 'features', label: 'FEATURES' },
  { id: 'ml', label: 'ML ANALYSIS' },
  { id: 'security', label: 'SECURITY' },
  { id: 'risk', label: 'RISK' },
  { id: 'result', label: 'RESULT' }
];

export default function ForensicPipeline({ activeFile, events }) {
  const [activeTab, setActiveTab] = useState('detect');
  const [autoScrollLog, setAutoScrollLog] = useState(true);
  const logEndRef = useRef(null);

  const isComp = checkIsComplete(activeFile);
  const isScan = stateFor(activeFile) === 'SCANNING';

  useEffect(() => {
    if (!activeFile) {
        setActiveTab('detect');
        return;
    }
    if (isComp) {
        setActiveTab('result');
    } else if (isScan) {
        setActiveTab('ml');
    } else {
        setActiveTab('detect');
    }
  }, [activeFile?.sha256, activeFile?.scan_status, activeFile?.verdict, activeFile?.vulnerability?.verdict]);

  useEffect(() => {
    if (autoScrollLog && logEndRef.current) {
        logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [events, autoScrollLog]);

  const getPhaseState = (id) => {
    if (!activeFile) return 'WAITING';
    if (id === 'detect' || id === 'identify') return 'COMPLETE';
    if (id === 'hash') {
        if (activeFile.sha256) return 'COMPLETE';
        if (isComp) return 'SKIPPED';
        return 'PROCESSING';
    }
    if (id === 'features' || id === 'ml') {
        if (isComp) return 'COMPLETE';
        return isScan ? 'PROCESSING' : 'WAITING';
    }
    if (id === 'security' || id === 'risk' || id === 'result') {
        if (isComp) return 'COMPLETE';
        return 'WAITING';
    }
    return 'WAITING';
  }

  const renderSubprocesses = (phaseId) => {
      if (!activeFile) return [];
      const f = activeFile;
      switch(phaseId) {
          case 'detect':
              return [
                  { name: 'Directory Event Received', state: 'COMPLETE', value: f.timestamp ? new Date(f.timestamp*1000).toLocaleTimeString() : '—' },
                  { name: 'File Path Captured', state: 'COMPLETE', value: f.path || '—' },
                  { name: 'File Existence Verified', state: 'COMPLETE' },
                  { name: 'File Metadata Captured', state: 'COMPLETE', value: f.size ? `${f.size} bytes` : '—' },
                  { name: 'Scan Job Created', state: 'COMPLETE' }
              ];
          case 'identify':
              return [
                  { name: 'Filename Parsed', state: 'COMPLETE', value: f.filename },
                  { name: 'Extension Identified', state: 'COMPLETE', value: f.filename?.split('.').pop() || '—' },
                  { name: 'File Type Validated', state: 'COMPLETE', value: f.file_type || '—' },
                  { name: 'MIME Type Detected', state: isComp ? 'UNAVAILABLE' : isScan ? 'PROCESSING' : 'WAITING' },
                  { name: 'File Signature Checked', state: isComp ? 'UNAVAILABLE' : isScan ? 'PROCESSING' : 'WAITING' },
              ];
          case 'hash':
              return [
                  { name: 'SHA-256 Generated', state: f.sha256 ? 'COMPLETE' : isScan ? 'PROCESSING' : 'WAITING', value: f.sha256 },
                  { name: 'MD5 Generated', state: isComp ? 'UNAVAILABLE' : isScan ? 'PROCESSING' : 'WAITING' },
                  { name: 'SHA-1 Generated', state: isComp ? 'UNAVAILABLE' : isScan ? 'PROCESSING' : 'WAITING' },
                  { name: 'Known-Hash Lookup', state: isComp ? 'SKIPPED' : isScan ? 'PROCESSING' : 'WAITING' },
              ];
          case 'features':
              const hasFeatures = !!f.vulnerability?.explanation?.top_features || !!f.explanation?.top_features;
              const fCount = f.vulnerability?.explanation?.top_features?.length || f.explanation?.top_features?.length;
              return [
                  { name: 'Metadata Extraction', state: hasFeatures ? 'COMPLETE' : isComp ? 'SKIPPED' : isScan ? 'PROCESSING' : 'WAITING' },
                  { name: 'Feature Vector Construction', state: hasFeatures ? 'COMPLETE' : isComp ? 'SKIPPED' : isScan ? 'PROCESSING' : 'WAITING', value: hasFeatures ? `${fCount} features extracted` : null },
                  { name: 'PE Header Parsing', state: isComp ? 'UNAVAILABLE' : isScan ? 'PROCESSING' : 'WAITING' },
                  { name: 'DOS Header Analysis', state: isComp ? 'UNAVAILABLE' : isScan ? 'PROCESSING' : 'WAITING' },
                  { name: 'Section Extraction', state: isComp ? 'UNAVAILABLE' : isScan ? 'PROCESSING' : 'WAITING' },
                  { name: 'Import Table Analysis', state: isComp ? 'UNAVAILABLE' : isScan ? 'PROCESSING' : 'WAITING' },
                  { name: 'Entropy Calculation', state: isComp ? 'UNAVAILABLE' : isScan ? 'PROCESSING' : 'WAITING' },
              ];
          case 'ml':
              const hasProb = f.vulnerability?.malware_probability != null || f.malware_probability != null;
              const prob = f.vulnerability?.malware_probability ?? f.malware_probability;
              const model = f.vulnerability?.model_version || f.model_version;
              return [
                  { name: 'Model Loaded', state: model ? 'COMPLETE' : isScan ? 'PROCESSING' : 'WAITING', value: model },
                  { name: 'Feature Preprocessing', state: hasProb ? 'COMPLETE' : isComp ? 'SKIPPED' : isScan ? 'PROCESSING' : 'WAITING' },
                  { name: 'Model Inference', state: hasProb ? 'COMPLETE' : isComp ? 'SKIPPED' : isScan ? 'PROCESSING' : 'WAITING', value: f.inference_ms ? `${f.inference_ms}ms` : null },
                  { name: 'Probability Calculation', state: hasProb ? 'COMPLETE' : isComp ? 'SKIPPED' : isScan ? 'PROCESSING' : 'WAITING', value: hasProb ? `${(prob*100).toFixed(2)}%` : null },
                  { name: 'Explainability', state: (f.vulnerability?.explanation || f.explanation) ? 'COMPLETE' : isComp ? 'UNAVAILABLE' : isScan ? 'PROCESSING' : 'WAITING' },
              ];
          case 'security':
              const hasVuln = !!f.vulnerability?.note || !!f.note;
              return [
                  { name: 'Structural Analysis', state: isComp ? 'UNAVAILABLE' : isScan ? 'PROCESSING' : 'WAITING' },
                  { name: 'Suspicious Characteristic Analysis', state: hasVuln ? 'COMPLETE' : isComp ? 'UNAVAILABLE' : isScan ? 'PROCESSING' : 'WAITING', value: f.vulnerability?.note || f.note },
                  { name: 'Security Finding Generation', state: isComp ? 'COMPLETE' : isScan ? 'PROCESSING' : 'WAITING', value: f.vulnerability?.risk_level || f.risk_level ? `Level: ${f.vulnerability?.risk_level || f.risk_level}` : null },
                  { name: 'Vulnerability Assessment', state: isComp ? 'COMPLETE' : isScan ? 'PROCESSING' : 'WAITING' },
              ];
          case 'risk':
              const hasRisk = f.vulnerability?.risk_score != null || f.risk_score != null;
              const score = f.vulnerability?.risk_score ?? f.risk_score;
              return [
                  { name: 'ML Confidence Received', state: hasRisk ? 'COMPLETE' : isComp ? 'SKIPPED' : 'WAITING' },
                  { name: 'Severity Factors Evaluated', state: hasRisk ? 'COMPLETE' : isComp ? 'SKIPPED' : 'WAITING' },
                  { name: 'Risk Score Calculation', state: hasRisk ? 'COMPLETE' : isComp ? 'SKIPPED' : 'WAITING', value: hasRisk ? `${score} / 100` : null },
              ];
          case 'result':
              const duration = f.total_scan_ms || f.analysis_duration_ms || f.vulnerability?.analysis_duration_ms;
              return [
                  { name: 'Malware Classification', state: isComp ? 'COMPLETE' : 'WAITING', value: stateFor(f) },
                  { name: 'Confidence Finalized', state: (f.vulnerability?.malware_probability != null || f.malware_probability != null) ? 'COMPLETE' : isComp ? 'SKIPPED' : 'WAITING' },
                  { name: 'Recommended Action', state: isComp ? 'COMPLETE' : 'WAITING', value: f.action || f.vulnerability?.action || '—' },
                  { name: 'Scan Completed', state: isComp ? 'COMPLETE' : 'WAITING', value: duration ? `${duration}ms` : null },
              ];
      }
      return [];
  }

  const renderIndicator = (st) => {
    switch(st) {
        case 'COMPLETE': return <div className="led led-green shrink-0" />;
        case 'PROCESSING': return <div className="led led-cyan animate-pulse shrink-0" />;
        case 'FAILED': return <div className="led led-red shrink-0" />;
        case 'UNAVAILABLE': return <div className="led led-orange shrink-0" />;
        case 'SKIPPED': return <div className="w-2 h-2 rounded-full bg-zinc-600 shrink-0" />;
        case 'WAITING': default: return <div className="led led-off shrink-0" />;
    }
  }

  const getStatusText = (st) => {
    switch(st) {
        case 'COMPLETE': return <span className="text-console-green">✓ COMPLETE</span>;
        case 'PROCESSING': return <span className="text-console-cyan">⟳ PROCESSING</span>;
        case 'FAILED': return <span className="text-console-red">× FAILED</span>;
        case 'UNAVAILABLE': return <span className="text-console-orange">— UNAVAILABLE</span>;
        case 'SKIPPED': return <span className="text-zinc-500">— SKIPPED</span>;
        case 'WAITING': default: return <span className="text-zinc-500">○ WAITING</span>;
    }
  }

  const subs = renderSubprocesses(activeTab);
  const completedSubs = subs.filter(s => s.state === 'COMPLETE').length;

  return (
    <div className="w-full bg-base-900 border border-black/80 rounded-xl overflow-hidden shadow-panel-raised flex flex-col">
      {/* Overall Scan Header */}
      <div className="bg-base-950 p-4 border-b border-white/5 flex items-center justify-between">
          <div>
              <div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase font-mono mb-1">
                  SCAN PROGRESS — PHASE {PHASES.findIndex(p => p.id === activeTab) + 1} / 8 — {PHASES.find(p => p.id === activeTab)?.label}
              </div>
              <div className="text-lg font-mono font-bold text-white tracking-wide truncate max-w-xl">
                  {activeFile ? activeFile.filename : 'NO FILE SELECTED'}
              </div>
          </div>
          {activeFile && (
              <div className="text-right">
                  <div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase font-mono mb-1">OVERALL STATUS</div>
                  <div className={`flex items-center gap-2 text-sm font-bold tracking-widest uppercase justify-end
                      ${isScan ? 'text-console-cyan' : stateFor(activeFile) === 'MALICIOUS' ? 'text-console-red' : 'text-console-green'}
                  `}>
                      <div className={`led ${isScan ? 'led-cyan animate-pulse' : stateFor(activeFile) === 'MALICIOUS' ? 'led-red' : 'led-green'}`} />
                      {stateFor(activeFile)}
                  </div>
              </div>
          )}
      </div>

      {/* Horizontal Phase Navigation */}
      <div className="flex overflow-x-auto border-b border-black/80 bg-base-800 p-2 gap-2 hide-scrollbar">
          {PHASES.map((phase, idx) => {
              const pState = getPhaseState(phase.id);
              const isActive = activeTab === phase.id;
              
              return (
                  <button 
                      key={phase.id}
                      onClick={() => setActiveTab(phase.id)}
                      className={`flex-1 min-w-[120px] p-3 rounded-lg border transition-all text-left relative overflow-hidden group
                          ${isActive ? 'bg-base-700 border-console-cyan/50 shadow-btn-pressed' : 'bg-base-900 border-black/80 shadow-btn-raised hover:bg-base-800'}
                      `}
                  >
                      {isActive && <div className="absolute top-0 left-0 w-1 h-full bg-console-cyan" />}
                      <div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase font-mono mb-2">PHASE {idx + 1}</div>
                      <div className="font-bold text-white text-xs mb-2 tracking-wide">{phase.label}</div>
                      <div className="flex items-center gap-2 text-[10px] font-mono">
                          {renderIndicator(pState)}
                          {getStatusText(pState)}
                      </div>
                  </button>
              )
          })}
      </div>

      {/* Subprocess Drawer */}
      <div className="p-6 bg-[#0a0c0e] shadow-panel-recessed min-h-[300px]">
          {!activeFile ? (
              <div className="h-full flex items-center justify-center text-zinc-500 font-mono text-sm opacity-50">
                  SCANNER READY — Waiting for files.
              </div>
          ) : (
              <div className="space-y-6">
                  <div className="flex justify-between items-end border-b border-white/5 pb-2">
                      <div className="text-sm font-bold text-white tracking-widest uppercase">{PHASES.find(p=>p.id===activeTab)?.label} SUBPROCESSES</div>
                      <div className="text-xs font-mono text-zinc-500">{completedSubs} / {subs.length} complete</div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-4">
                      {subs.map((s, idx) => (
                          <div key={idx} className="flex flex-col gap-1">
                              <div className="flex items-start justify-between">
                                  <div className="flex items-center gap-3">
                                      {renderIndicator(s.state)}
                                      <span className={`text-xs font-mono font-bold ${s.state === 'COMPLETE' ? 'text-zinc-200' : s.state === 'PROCESSING' ? 'text-console-cyan' : 'text-zinc-500'}`}>
                                          {s.name}
                                      </span>
                                  </div>
                                  <div className="text-[10px] font-mono">{getStatusText(s.state)}</div>
                              </div>
                              {s.value && (
                                  <div className="ml-5 text-xs font-mono text-zinc-400 mt-1 pl-2 border-l border-white/10 break-all">
                                      {s.value}
                                  </div>
                              )}
                          </div>
                      ))}
                  </div>
              </div>
          )}
      </div>

      {/* LIVE ANALYSIS LOG */}
      <div className="border-t border-black/80 bg-base-950 flex flex-col">
          <div className="flex justify-between items-center px-4 py-2 border-b border-white/5 bg-base-900">
              <div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase font-mono flex items-center gap-2">
                  <div className="led led-green animate-pulse" /> LIVE ANALYSIS LOG
              </div>
              <button 
                  onClick={() => setAutoScrollLog(!autoScrollLog)}
                  className={`text-[10px] font-mono px-2 py-1 rounded border ${autoScrollLog ? 'bg-base-700 border-console-cyan/30 text-console-cyan' : 'bg-base-800 border-black text-zinc-500'}`}
              >
                  {autoScrollLog ? 'AUTO-SCROLL ON' : 'AUTO-SCROLL OFF'}
              </button>
          </div>
          <div className="h-[200px] overflow-y-auto p-4 font-mono text-xs space-y-2">
              {events.length === 0 ? (
                  <div className="text-zinc-600">Waiting for live events...</div>
              ) : (
                  [...events].reverse().map((e, idx) => (
                      <div key={idx} className="flex gap-4">
                          <span className="text-zinc-600 shrink-0">
                              {new Date(e.time * 1000).toISOString().split('T')[1].slice(0, -1)}
                          </span>
                          <span className={`
                              ${e.type === 'alert' ? 'text-console-red' : 
                                e.type === 'detected' ? 'text-console-blue' : 
                                e.type === 'quarantine' ? 'text-console-orange' : 
                                e.type === 'scanning' ? 'text-console-cyan' : 'text-zinc-300'}
                          `}>
                              [{e.type.toUpperCase()}] {e.msg}
                          </span>
                      </div>
                  ))
              )}
              <div ref={logEndRef} />
          </div>
      </div>
    </div>
  )
}
