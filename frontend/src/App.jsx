import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import NetworkNodes from './components/NetworkNodes'
import { BentoGrid } from './components/BentoCard'
import BentoCard from './components/BentoCard'

function OverviewView() {
  const [prot, setProt] = useState(null)
  const [health, setHealth] = useState(null)
  useEffect(() => {
    fetch('/api/v1/protection/status').then(r=>r.json()).then(setProt).catch(()=>{})
    fetch('/health').then(r=>r.json()).then(setHealth).catch(()=>{})
  }, [])
  return (
    <BentoGrid className="grid-cols-4 auto-rows-[130px]">
      <BentoCard span="col-span-2 row-span-2" eyebrow="System" title="Health Overview" glow="cyan" icon={<span className="text-cyan-400">●</span>}>
        <div className="flex flex-col h-full justify-between">
          <div className="grid grid-cols-3 gap-3">
            <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] p-3">
              <div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase">Active Model</div>
              <div className="text-sm font-mono font-semibold text-white mt-1 truncate">{prot?.active_model || 'mlp-central-v1'}</div>
              <div className="text-xs text-emerald-400 mt-1">● {prot?.protection || 'active'}</div>
            </div>
            <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] p-3">
              <div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase">Global F1</div>
              <div className="text-2xl font-black text-white mt-1">0.92</div>
              <div className="text-xs text-zinc-500">+2.1% vs last round</div>
            </div>
            <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] p-3">
              <div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase">Rounds</div>
              <div className="text-2xl font-black text-white mt-1">42</div>
              <div className="text-xs text-zinc-500">30 clients · fedavg</div>
            </div>
          </div>
          <div className="mt-3 flex items-center gap-2 text-xs text-zinc-500">
            <span className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]" /> {health?.host || '127.0.0.1'}:{health?.port || 8080} · {health?.secure ? 'Secure' : 'Plain'}
          </div>
        </div>
      </BentoCard>
      <BentoCard span="col-span-1" eyebrow="Threats" title="Detections" glow="violet">
        <div className="text-3xl font-black">12</div>
        <div className="text-xs text-zinc-500">quarantined 3 · last 24h</div>
        <div className="mt-2 h-1.5 rounded-full bg-white/10 overflow-hidden"><div className="h-full bg-violet-500" style={{width:'68%'}}/></div>
      </BentoCard>
      <BentoCard span="col-span-1" eyebrow="Live" title="Monitor" glow="cyan">
        <div className="text-sm font-semibold text-emerald-400">● Running</div>
        <div className="text-xs text-zinc-500 mt-1">D:/Telegram · D:/Downloads</div>
        <div className="text-xs text-zinc-400 mt-2">2 files in last hour</div>
      </BentoCard>
      <BentoCard span="col-span-1" eyebrow="Resource" title="CPU 42%" glow="neutral">
        <div className="h-1.5 rounded-full bg-white/10 overflow-hidden mt-2"><div className="h-full bg-cyan-400" style={{width:'42%'}}/></div>
        <div className="text-xs text-zinc-500 mt-2">Training idle · policy OK</div>
      </BentoCard>
      <BentoCard span="col-span-1" eyebrow="Drift" title="No drift" glow="neutral">
        <div className="text-sm font-semibold text-emerald-400">● Stable</div>
        <div className="text-xs text-zinc-500 mt-1">PSI 0.06 · threshold 0.2</div>
      </BentoCard>
      <BentoCard span="col-span-2" eyebrow="Federated" title="Global vs Local F1" glow="violet">
        <div className="flex gap-2 mt-1">
          <span className="px-2 py-1 rounded-full text-xs font-bold bg-cyan-500/15 text-cyan-300 border border-cyan-500/20">Global 0.92</span>
          <span className="px-2 py-1 rounded-full text-xs font-bold bg-violet-500/15 text-violet-300 border border-violet-500/20">Mean local 0.88</span>
          <span className="px-2 py-1 rounded-full text-xs bg-white/5 text-zinc-400 border border-white/10">Worst 0.81</span>
        </div>
        <div className="text-xs text-zinc-500 mt-3">Non-IID moderate · 10 clients</div>
      </BentoCard>
      <BentoCard span="col-span-2" eyebrow="Compliance" title="Privacy & Audit" glow="neutral">
        <div className="text-xs text-zinc-400 leading-relaxed">DP disabled · audit log enabled · retention 0 days · No raw file uploads — telemetry only.</div>
      </BentoCard>
    </BentoGrid>
  )
}

function TrainingView() {
  return (
    <BentoGrid className="grid-cols-4 auto-rows-[130px]">
      <BentoCard span="col-span-2 row-span-2" eyebrow="FL Engine" title="Training Progress" glow="cyan" icon={<span className="text-cyan-400">⬡</span>}>
        <div className="h-full flex flex-col justify-between">
          <div className="grid grid-cols-3 gap-3">
            <div><div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase">Algorithm</div><div className="text-sm font-semibold text-white mt-1">FedAvg</div></div>
            <div><div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase">Round</div><div className="text-xl font-black text-white">18/30</div></div>
            <div><div className="text-[10px] tracking-widest font-bold text-zinc-500 uppercase">Clients</div><div className="text-xl font-black text-white">10</div></div>
          </div>
          <div className="space-y-2">
            <div className="h-2 rounded-full bg-white/10 overflow-hidden"><div className="h-full bg-cyan-400" style={{width:'60%'}}/></div>
            <div className="text-xs text-zinc-500">Global F1 0.92 · next round in 12s</div>
          </div>
        </div>
      </BentoCard>
      <BentoCard span="col-span-2" eyebrow="Model" title="Registry" glow="violet">
        <div className="flex justify-between items-center">
          <div><div className="text-sm font-mono font-semibold">mlp-central-v1</div><div className="text-xs text-zinc-500">centralized · 2381 in · 256/128</div></div>
          <span className="px-2 py-1 rounded-full text-xs font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/20">ACTIVE</span>
        </div>
      </BentoCard>
      <BentoCard span="col-span-1" eyebrow="Drift" title="PSI 0.06" glow="neutral">
        <div className="text-sm font-semibold text-emerald-400">No drift</div>
        <div className="text-xs text-zinc-500">Stable distribution</div>
      </BentoCard>
      <BentoCard span="col-span-1" eyebrow="Compare" title="FedProx Δ" glow="neutral">
        <div className="text-xl font-black">+1.2%</div>
        <div className="text-xs text-zinc-500">vs FedAvg</div>
      </BentoCard>
      <BentoCard span="col-span-2" eyebrow="Non-IID" title="Partition Health" glow="cyan">
        <div className="text-xs text-zinc-400">Strategy <b className="text-white">moderate</b> · Dirichlet 0.5 · worst client F1 0.81 · variance 0.012</div>
        <div className="mt-2 h-1.5 rounded-full bg-white/10 overflow-hidden"><div className="h-full bg-violet-400" style={{width:'72%'}}/></div>
      </BentoCard>
      <BentoCard span="col-span-2" eyebrow="Logs" title="Recent Rounds" glow="neutral">
        <div className="text-xs font-mono text-zinc-400 space-y-1">
          <div>R18 · acc 0.93 · F1 0.92 · 1.1 MB</div>
          <div>R17 · acc 0.92 · F1 0.91 · 1.1 MB</div>
          <div>R16 · acc 0.91 · F1 0.90 · 1.0 MB</div>
        </div>
      </BentoCard>
    </BentoGrid>
  )
}

function SecurityView() {
  return (
    <BentoGrid className="grid-cols-4 auto-rows-[130px]">
      <BentoCard span="col-span-2 row-span-2" eyebrow="Live Scan" title="File Arrivals" glow="cyan" icon={<span className="text-cyan-400">⬢</span>}>
        <div className="space-y-2">
          <div className="flex justify-between text-xs"><span className="text-zinc-400">D:/Telegram/live-pe.exe</span><span className="px-2 py-0.5 rounded-full text-[11px] font-bold bg-amber-500/15 text-amber-300 border border-amber-500/20">MEDIUM</span></div>
          <div className="flex justify-between text-xs"><span className="text-zinc-400">D:/Telegram/live-test.txt</span><span className="px-2 py-0.5 rounded-full text-[11px] font-bold bg-zinc-800 text-zinc-400 border border-white/10">skipped</span></div>
          <div className="text-xs text-zinc-500 mt-2">2 files now · every arrival visible · 7-step pipeline</div>
        </div>
      </BentoCard>
      <BentoCard span="col-span-1 row-span-2" eyebrow="Quarantine" title="3 files" glow="violet">
        <div className="space-y-2 text-xs">
          <div className="flex justify-between"><span>demo.exe</span><span className="text-amber-400">WARN</span></div>
          <div className="flex justify-between"><span>payload.dll</span><span className="text-red-400">HIGH</span></div>
          <div className="flex justify-between"><span>stub.exe</span><span className="text-zinc-500">ERROR</span></div>
        </div>
      </BentoCard>
      <BentoCard span="col-span-1" eyebrow="Risk" title="Vulnerability" glow="cyan">
        <div className="text-2xl font-black">49<span className="text-sm text-zinc-500">/100</span></div>
        <div className="text-xs text-amber-300">MEDIUM · top: header_timestamp</div>
      </BentoCard>
      <BentoCard span="col-span-2" eyebrow="Privacy" title="DP & Compliance" glow="neutral">
        <div className="text-xs text-zinc-400 leading-relaxed">No raw uploads · SHA + probability + risk only · DP disabled · audit 1000 entries</div>
      </BentoCard>
      <BentoCard span="col-span-2" eyebrow="Timeline" title="Scan Pipeline" glow="neutral">
        <div className="flex flex-wrap gap-1.5">
          {["stable","sha256","features","inference","risk","verdict","quarantine"].map(s=> <span key={s} className="px-2 py-1 rounded-full text-[11px] font-semibold bg-white/5 border border-white/10 text-zinc-300">{s}</span>)}
        </div>
      </BentoCard>
    </BentoGrid>
  )
}

function NetworkView() { return <NetworkNodes /> }

const VIEWS = {
  overview: { label: 'Overview', Icon: '◧', Comp: OverviewView },
  training: { label: 'Training & Models', Icon: '⬡', Comp: TrainingView },
  network: { label: 'Network & Nodes', Icon: '⬢', Comp: NetworkView },
  security: { label: 'Security & Compliance', Icon: '⬔', Comp: SecurityView },
}

export default function App() {
  const [active, setActive] = useState('overview')
  const ActiveComp = VIEWS[active].Comp

  return (
    <div className="min-h-screen bg-base-950 text-white relative">
      <div className="bg-field" aria-hidden />
      <header className="sticky top-0 z-20 backdrop-blur-xl bg-base-950/60 border-b border-white/[0.06]">
        <div className="max-w-[1280px] mx-auto px-6 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-white/[0.06] border border-white/10 grid place-items-center font-black tracking-widest text-sm">FS</div>
            <div>
              <h1 className="text-[15px] font-bold tracking-tight leading-none">FedShield</h1>
              <p className="text-[11px] text-zinc-400 tracking-wide">Control Center · Glass Bento</p>
            </div>
            <span className="hidden md:inline-flex ml-3 px-2.5 py-1 rounded-full text-[10px] font-bold tracking-widest bg-white/10 border border-white/10">DARK · GLASS</span>
          </div>
          <div className="hidden md:flex items-center gap-2 text-xs text-zinc-400">
            <span className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.8)]" /> Live
          </div>
        </div>
        <div className="max-w-[1280px] mx-auto px-6 pb-4">
          <nav className="mx-auto w-fit flex items-center gap-1 p-1.5 rounded-full glass-strong shadow-glass">
            {Object.entries(VIEWS).map(([key, { label, Icon }]) => {
              const isActive = active === key
              return (
                <button key={key} onClick={() => setActive(key)} className={`relative px-4 py-2 rounded-full text-[13px] font-semibold tracking-tight transition flex items-center gap-2 ${isActive ? 'text-white' : 'text-zinc-400 hover:text-zinc-200'}`}>
                  {isActive && <motion.div layoutId="pill" className="absolute inset-0 rounded-full bg-white/[0.10] border border-white/10 shadow-glow-cyan" transition={{ type: 'spring', stiffness: 400, damping: 30 }} />}
                  <span className="relative text-[13px] opacity-80">{Icon}</span>
                  <span className="relative hidden sm:inline">{label}</span>
                  <span className="relative sm:hidden">{label.split(' ')[0]}</span>
                </button>
              )
            })}
          </nav>
        </div>
      </header>
      <main className="max-w-[1280px] mx-auto px-6 py-6">
        <AnimatePresence mode="wait">
          <motion.div key={active} initial={{ opacity: 0, y: 8, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -8, scale: 0.98 }} transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}>
            <div className="mb-3 flex items-center gap-2">
              <h2 className="text-[11px] font-bold tracking-[0.14em] text-zinc-500 uppercase">{VIEWS[active].label}</h2>
              <span className="h-px flex-1 bg-white/[0.06]" />
              <span className="text-[11px] text-zinc-500">4 views · Bento</span>
            </div>
            <ActiveComp />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}
