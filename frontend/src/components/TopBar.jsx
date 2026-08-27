import React from 'react'
export default function TopBar({ page, system }) {
  return (
    <div className="sticky top-0 z-20 backdrop-blur-xl bg-[#0a0a0a]/70 border-b border-white/[0.06] shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
      <div className="flex items-center justify-between gap-4 px-6 py-3">
        <div className="flex items-center gap-4">
          <div className="hidden md:block text-[11px] font-black tracking-[0.14em] text-zinc-500 uppercase">FEDSHIELD</div>
          <span className="hidden md:block w-px h-4 bg-white/10" />
          <div className="text-[13px] font-semibold tracking-tight text-white">{page}</div>
          <span className="hidden sm:inline-flex items-center gap-1.5 ml-2 px-2.5 py-1 rounded-full text-[11px] font-bold bg-white/[0.06] border border-white/10 text-zinc-300">
            <span className={`w-1.5 h-1.5 rounded-full ${system?.online ? 'bg-emerald-400 shadow-[0_0_6px_rgba(16,185,129,0.8)]' : 'bg-amber-400'}`} /> SYSTEM {system?.online ? 'ONLINE' : 'OFFLINE'}
          </span>
          <span className="hidden lg:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold bg-emerald-500/10 border border-emerald-500/20 text-emerald-300">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(16,185,129,0.8)]" /> MONITORING ACTIVE
          </span>
        </div>
        <div className="flex items-center gap-3">
          <div className="hidden md:flex items-center gap-2 text-[11px] font-mono text-zinc-500">
            <span className={`inline-flex items-center gap-1.5 ${system?.backend === 'CONNECTED' ? 'text-emerald-400' : 'text-amber-400'}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${system?.backend === 'CONNECTED' ? 'bg-emerald-400' : 'bg-amber-400'}`} /> {system?.backend || 'CONNECTING'}
            </span>
          </div>
          <button className="w-8 h-8 rounded-full bg-white/[0.06] border border-white/10 grid place-items-center text-zinc-400 hover:text-white hover:bg-white/10 transition shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]">◯</button>
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-cyan-500 border border-white/10 shadow-[0_2px_8px_rgba(0,0,0,0.4)] grid place-items-center text-[11px] font-black text-white">AD</div>
        </div>
      </div>
    </div>
  )
}
