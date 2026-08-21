"""Resource service (Phase 20 §5).

Expose permitted client resource summaries: training status, policy state,
aggregated metrics. Avoid detailed personal telemetry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ResourceService:
    def __init__(self):
        self._aggregated: Dict[str, Any] = {"training_status": "idle", "policy": {}, "metrics": {}}
        self._per_client: Dict[str, Dict[str, Any]] = {}

    def update_client_resource(self, client_id: str, payload: Dict[str, Any]) -> None:
        # Only store aggregated fields, drop detailed personal telemetry
        allowed = {"training_status", "policy_state", "cpu_percent", "memory_mb", "battery_percent"}
        filtered = {k: v for k, v in payload.items() if k in allowed}
        self._per_client[client_id] = filtered
        self._recompute_aggregated()

    def _recompute_aggregated(self) -> None:
        if not self._per_client:
            return
        # Aggregate without exposing per-client details
        statuses = [v.get("training_status") for v in self._per_client.values()]
        self._aggregated["training_status"] = "training" if "training" in statuses else "idle"
        # Average CPU/memory if present, but not per-client
        cpus = [v["cpu_percent"] for v in self._per_client.values() if "cpu_percent" in v]
        if cpus:
            self._aggregated["metrics"]["avg_cpu_percent"] = sum(cpus) / len(cpus)
        self._aggregated["active_clients"] = len(self._per_client)

    def get_status(self) -> Dict[str, Any]:
        return {"training_status": self._aggregated.get("training_status", "idle")}

    def get_policy(self) -> Dict[str, Any]:
        return self._aggregated.get("policy", {})

    def get_metrics(self) -> Dict[str, Any]:
        # Return aggregated only, no per-client detailed telemetry
        return self._aggregated.get("metrics", {})
