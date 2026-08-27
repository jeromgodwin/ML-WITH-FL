// API client for FedShield Control Center — no file-upload, only telemetry summaries
const BASE = ''

async function fetchJSON(path, opts = {}) {
  const res = await fetch(BASE + path, { headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) }, ...opts })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status} ${text}`)
  }
  return res.json()
}

export const api = {
  health: () => fetchJSON('/health'),
  serverStatus: () => fetchJSON('/api/v1/status'),
  protectionStatus: () => fetchJSON('/api/v1/protection/status'),
  activeModel: (headers) => fetchJSON('/api/v1/model/active', { headers }),
  clients: (headers) => fetchJSON('/api/v1/clients', { headers }),
  detections: (limit = 20, headers) => fetchJSON(`/api/v1/detections?limit=${limit}`, { headers }),
  flAlgorithms: () => fetchJSON('/api/v1/fl/algorithms'),
  flComparison: () => fetchJSON('/api/v1/fl/comparison'),
  resourceStatus: () => fetchJSON('/api/v1/resource/status'),
  resourceMetrics: () => fetchJSON('/api/v1/resource/metrics'),
  driftStatus: () => fetchJSON('/api/v1/drift/status'),
  driftLastEvent: () => fetchJSON('/api/v1/drift/last_event'),
  models: () => fetchJSON('/api/v1/models'),
  candidates: () => fetchJSON('/api/v1/models/candidates'),
  communication: (expId) => fetchJSON(`/api/v1/fl/experiments/${expId}/metrics/rounds`),
  monitorStatus: () => fetchJSON('/api/v1/monitor/status'),
  monitorFiles: (limit = 20) => fetchJSON(`/api/v1/monitor/files?limit=${limit}`),
  monitorDetections: (limit = 20) => fetchJSON(`/api/v1/monitor/detections?limit=${limit}`),
}
