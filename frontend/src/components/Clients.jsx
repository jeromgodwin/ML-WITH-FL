import React, { useEffect, useState } from 'react'

export default function Clients() {
  const [clients, setClients] = useState([])
  const [err, setErr] = useState(null)

  useEffect(() => {
    fetch('/api/v1/clients', { headers: { 'X-Client-Id': 'admin-1', Authorization: 'Bearer dummy' } })
      .then(r => r.ok ? r.json() : Promise.reject(r.statusText))
      .then(setClients)
      .catch(e => setErr(String(e)))
  }, [])

  return (
    <div className="card">
      <h3>Clients — no raw files exposed</h3>
      {err && <div className="badge err">{err} (auth required)</div>}
      <table>
        <thead><tr><th>Client ID</th><th>Status</th><th>Last Seen</th><th>Active Model</th><th>FL Participation</th><th>Resource/Training</th></tr></thead>
        <tbody>
          {clients.map(c => (
            <tr key={c.client_id}>
              <td>{c.client_id}</td>
              <td><span className={`badge ${c.connection_status === 'online' ? 'ok' : 'warn'}`}>{c.connection_status}</span></td>
              <td>{c.last_seen ? new Date(c.last_seen * 1000).toLocaleString() : '—'}</td>
              <td>{c.active_model || '—'}</td>
              <td>{c.last_fl_participation ? new Date(c.last_fl_participation * 1000).toLocaleString() : '—'}</td>
              <td>{c.resource_state || '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {clients.length === 0 && !err && <div>No clients yet</div>}
    </div>
  )
}
