import React, { useEffect, useState } from 'react'
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts'
import { Radio, Cpu, Gauge, ArrowLeftRight } from 'lucide-react'
import { BentoGrid } from '../components/BentoCard'
import BentoCard from '../components/BentoCard'
import { api } from '../api'

export default function NetworkNodes() {
  const [clients, setClients] = useState([])
  const [clientsErr, setClientsErr] = useState(null)
  const [protection, setProtection] = useState(null)
  const [resourceStatus, setResourceStatus] = useState(null)
  const [resourceMetrics, setResourceMetrics] = useState(null)
  const [comparison, setComparison] = useState([])

  useEffect(() => {
    api.clients({ 'X-Client-Id': 'admin-1', Authorization: 'Bearer dummy' })
      .then(setClients)
      .catch((e) => setClientsErr(String(e)))
    api.protectionStatus().then(setProtection).catch(() => {})
    api.resourceStatus().then(setResourceStatus).catch(() => {})
    api.resourceMetrics().then(setResourceMetrics).catch(() => {})
    api.flComparison().then((d) => setComparison(Array.isArray(d) ? d : [])).catch(() => {})
  }, [])

  const online = clients.filter((c) => c.connection_status === 'online').length

  const commsSeries = toCommsSeries(comparison)
  const nodeLoadSeries = toNodeLoad(clients)

  return (
    <BentoGrid className="grid-cols-4 auto-rows-[130px]">
      {/* Active nodes — hero tile */}
      <BentoCard
        span="col-span-1 row-span-2"
        eyebrow="Fleet"
        title="Active Nodes"
        icon={<Radio size={16} />}
        glow="cyan"
      >
        <div className="flex h-full flex-col justify-between">
          <div>
            <div className="font-display text-4xl font-semibold text-white tabular-nums">
              {online}
              <span className="text-lg text-zinc-500">/{clients.length || '—'}</span>
            </div>
            <div className="mt-1 text-xs text-zinc-500">online / registered</div>
          </div>
          {clientsErr && (
            <div className="text-[11px] text-amber-400/80 font-mono">auth required</div>
          )}
        </div>
      </BentoCard>

      {/* Node table — wide */}
      <BentoCard
        span="col-span-3 row-span-2"
        eyebrow="Clients"
        title="Registered Endpoints"
        icon={<Cpu size={16} />}
        glow="violet"
        interactive={false}
      >
        <NodeTable clients={clients} error={clientsErr} />
      </BentoCard>

      {/* Latency / round-trip chart */}
      <BentoCard
        span="col-span-2"
        eyebrow="Comms"
        title="Bytes per Round"
        icon={<ArrowLeftRight size={16} />}
        glow="cyan"
      >
        <MiniAreaChart data={commsSeries} dataKey="bytes" color="cyan" />
      </BentoCard>

      {/* Resource load chart */}
      <BentoCard
        span="col-span-1"
        eyebrow="Load"
        title="CPU / Mem"
        icon={<Gauge size={16} />}
        glow="violet"
      >
        <MiniBarChart data={nodeLoadSeries} />
      </BentoCard>

      {/* Training / resource status */}
      <BentoCard span="col-span-1" eyebrow="FL Engine" title="Training Status" glow="neutral">
        <div className="flex h-full flex-col justify-center gap-1.5">
          <StatusPill value={resourceStatus?.training_status || 'idle'} />
          {(resourceStatus?.reason || resourceMetrics?.reason) && (
            <div className="text-[11px] text-zinc-500 leading-snug">
              {resourceStatus?.reason || resourceMetrics?.reason}
            </div>
          )}
        </div>
      </BentoCard>

      {/* Endpoint protection strip */}
      <BentoCard
        span="col-span-4"
        eyebrow="Endpoint Protection"
        title="Aggregate Detection Telemetry"
        glow="neutral"
        interactive={false}
      >
        <div className="grid grid-cols-4 h-full items-center gap-4">
          <Stat label="Protection" value={protection?.protection || 'active'} />
          <Stat label="Active model" value={protection?.active_model || '—'} mono />
          <Stat label="Detections" value={protection?.detections_count ?? '—'} />
          <Stat label="Quarantined" value={protection?.quarantined_count ?? '—'} />
        </div>
      </BentoCard>
    </BentoGrid>
  )
}

function NodeTable({ clients, error }) {
  if (error) {
    return <div className="text-sm text-amber-400/80">{error} (auth required)</div>
  }
  if (clients.length === 0) {
    return <div className="text-sm text-zinc-500">No clients yet</div>
  }
  return (
    <div className="h-full overflow-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-[11px] uppercase tracking-wide text-zinc-500">
            <th className="pb-2 font-normal">Client</th>
            <th className="pb-2 font-normal">Status</th>
            <th className="pb-2 font-normal">Last seen</th>
            <th className="pb-2 font-normal">Model</th>
          </tr>
        </thead>
        <tbody className="text-zinc-300">
          {clients.map((c) => (
            <tr key={c.client_id} className="border-t border-white/5">
              <td className="py-1.5 font-mono text-xs text-zinc-400">{c.client_id}</td>
              <td className="py-1.5">
                <StatusPill value={c.connection_status} compact />
              </td>
              <td className="py-1.5 text-xs text-zinc-500">
                {c.last_seen ? new Date(c.last_seen * 1000).toLocaleTimeString() : '—'}
              </td>
              <td className="py-1.5 text-xs text-zinc-500">{c.active_model || '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function MiniAreaChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
        <defs>
          <linearGradient id="cyanFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.45} />
            <stop offset="100%" stopColor="#22d3ee" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
        <XAxis dataKey="round" tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false} width={36} />
        <Tooltip content={<GlassTooltip />} />
        <Area
          type="monotone"
          dataKey="bytes"
          stroke="#22d3ee"
          strokeWidth={2}
          fill="url(#cyanFill)"
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}

function MiniBarChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 4, right: 0, left: -28, bottom: 0 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
        <XAxis dataKey="node" tick={{ fill: '#71717a', fontSize: 9 }} axisLine={false} tickLine={false} />
        <YAxis hide />
        <Tooltip content={<GlassTooltip />} />
        <Bar dataKey="load" fill="#a855f7" radius={[3, 3, 0, 0]} maxBarSize={14} />
      </BarChart>
    </ResponsiveContainer>
  )
}

function GlassTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg border border-white/10 bg-base-900/90 backdrop-blur-md px-2.5 py-1.5 text-xs shadow-glass">
      <div className="text-zinc-500 font-mono">{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="text-zinc-200 tabular-nums">
          {p.value?.toLocaleString?.() ?? p.value}
        </div>
      ))}
    </div>
  )
}

function StatusPill({ value, compact = false }) {
  const ok = ['online', 'active', 'training', 'idle'].includes(value)
  const warn = ['paused', 'degraded'].includes(value)
  const color = warn ? 'amber' : ok ? 'emerald' : 'zinc'
  const dot = {
    emerald: 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]',
    amber: 'bg-amber-400 shadow-[0_0_6px_rgba(251,191,36,0.8)]',
    zinc: 'bg-zinc-500',
  }[color]
  const text = {
    emerald: 'text-emerald-300',
    amber: 'text-amber-300',
    zinc: 'text-zinc-400',
  }[color]

  return (
    <span className={`inline-flex items-center gap-1.5 ${text} ${compact ? 'text-xs' : 'text-sm font-medium'}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${dot}`} />
      {value}
    </span>
  )
}

function Stat({ label, value, mono = false }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={`mt-0.5 text-sm text-zinc-200 ${mono ? 'font-mono' : 'font-medium'}`}>{value}</div>
    </div>
  )
}

function toCommsSeries(comparison) {
  if (!comparison?.length) {
    return Array.from({ length: 8 }, (_, i) => ({ round: `R${i + 1}`, bytes: 0 }))
  }
  return comparison.map((c, i) => ({
    round: c.experiment_id ? String(c.experiment_id).slice(-4) : `R${i + 1}`,
    bytes: c.total_bytes && c.rounds ? Math.round(c.total_bytes / c.rounds) : c.bytes_per_round || 0,
  }))
}

function toNodeLoad(clients) {
  if (!clients?.length) {
    return Array.from({ length: 5 }, (_, i) => ({ node: `n${i + 1}`, load: 0 }))
  }
  return clients.slice(0, 6).map((c, i) => ({
    node: c.client_id ? c.client_id.slice(-4) : `n${i + 1}`,
    load: resourceStateToLoad(c.resource_state),
  }))
}

function resourceStateToLoad(state) {
  const map = { idle: 15, training: 70, throttled: 45, offline: 0 }
  return map[state] ?? 30
}
