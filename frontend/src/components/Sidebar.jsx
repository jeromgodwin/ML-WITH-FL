import React from 'react'

export default function Sidebar({ active, onSelect, system }) {
  const items = [
    { id: 'overview', label: 'Overview' },
    { id: 'monitor', label: 'Live Monitor' },
    { id: 'analysis', label: 'File Analysis' },
    { id: 'threats', label: 'Threats' },
    { id: 'federated', label: 'Federated Learning' },
    { id: 'experiments', label: 'Experiments' },
    { id: 'reports', label: 'Reports' },
    { id: 'settings', label: 'Settings' },
  ]

  return (
    <aside className="w-64 flex-shrink-0 border-r border-black/80 bg-base-900 shadow-[2px_0_10px_rgba(0,0,0,0.5)] flex flex-col z-10 relative">
      <div className="absolute top-0 right-0 bottom-0 w-px bg-white/5" />
      
      <div className="p-6">
        <h1 className="font-display text-xl font-bold tracking-widest text-zinc-100 drop-shadow-md">
          FEDSHIELD
        </h1>
        <div className="mt-1 font-mono text-[9px] uppercase tracking-widest text-zinc-500 font-bold">
          SECURITY INTELLIGENCE CONSOLE
        </div>
      </div>

      <nav className="flex-1 px-4 space-y-2 overflow-y-auto">
        {items.map((item) => {
          const isActive = active === item.id
          return (
            <button
              key={item.id}
              onClick={() => onSelect(item.id)}
              className={`
                w-full text-left px-4 py-3 rounded-lg text-sm font-semibold transition-all duration-200
                ${isActive 
                  ? 'bg-[#060809] shadow-panel-recessed text-cyan-400 border border-black/80' 
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.02]'
                }
              `}
            >
              <div className="flex items-center gap-3">
                <div className={`led ${isActive ? 'led-cyan' : 'led-off opacity-20'}`} />
                {item.label}
              </div>
            </button>
          )
        })}
      </nav>

      <div className="p-5 border-t border-black/80 bg-base-950/50 shadow-[0_-1px_0_rgba(255,255,255,0.02)] space-y-3">
        <div className="flex items-center justify-between text-xs font-mono font-bold">
          <span className="text-zinc-500">SYSTEM</span>
          <span className="flex items-center gap-2 text-zinc-300">
            {system.online ? 'ONLINE' : 'OFFLINE'}
            <div className={`led ${system.online ? 'led-green' : 'led-red'}`} />
          </span>
        </div>
        <div className="flex items-center justify-between text-xs font-mono font-bold">
          <span className="text-zinc-500">MONITOR</span>
          <span className="flex items-center gap-2 text-zinc-300">
            ACTIVE
            <div className="led led-blue" />
          </span>
        </div>
        <div className="flex items-center justify-between text-xs font-mono font-bold">
          <span className="text-zinc-500">BACKEND</span>
          <span className="flex items-center gap-2 text-zinc-300">
            {system.backend}
            <div className={`led ${system.backend === 'CONNECTED' ? 'led-cyan' : system.backend === 'DISCONNECTED' ? 'led-red' : 'led-orange'}`} />
          </span>
        </div>
      </div>
    </aside>
  )
}
