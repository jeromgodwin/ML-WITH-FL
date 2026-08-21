"""Client management service (Phase 20 §2).

Registered clients, client ID, connection status, last-seen, active model,
last FL participation. Does not expose sensitive local client data.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.federated.network.auth import ClientRegistry
from src.federated.model_registry import ModelRegistry


class ClientService:
    def __init__(self, client_registry: Optional[ClientRegistry] = None, model_registry: Optional[ModelRegistry] = None):
        self.client_registry = client_registry or ClientRegistry("data/server_registry")
        self.model_registry = model_registry or ModelRegistry("data/server_registry")
        # In-memory last-seen and FL participation (persisted minimally)
        self._last_seen: Dict[str, float] = {}
        self._last_fl: Dict[str, float] = {}

    def list_clients(self) -> List[Dict[str, Any]]:
        clients = self.client_registry.list_clients()
        out = []
        for cid, data in clients.items():
            # Do not expose token
            out.append({
                "client_id": cid,
                "role": data.get("role"),
                "connection_status": "online" if time.time() - self._last_seen.get(cid, 0) < 300 else "offline",
                "last_seen": self._last_seen.get(cid),
                "active_model": self._active_model_for_client(cid),
                "last_fl_participation": self._last_fl.get(cid),
            })
        return out

    def get_client(self, client_id: str) -> Optional[Dict[str, Any]]:
        data = self.client_registry.get(client_id)
        if not data:
            return None
        return {
            "client_id": client_id,
            "role": data.role,
            "connection_status": "online" if time.time() - self._last_seen.get(client_id, 0) < 300 else "offline",
            "last_seen": self._last_seen.get(client_id),
            "active_model": self._active_model_for_client(client_id),
            "last_fl_participation": self._last_fl.get(client_id),
        }

    def mark_seen(self, client_id: str) -> None:
        self._last_seen[client_id] = time.time()

    def mark_fl_participation(self, client_id: str) -> None:
        self._last_fl[client_id] = time.time()
        self.mark_seen(client_id)

    def _active_model_for_client(self, client_id: str) -> Optional[str]:
        try:
            a = self.model_registry.get_active()
            return a.version if a else None
        except Exception:
            return None
