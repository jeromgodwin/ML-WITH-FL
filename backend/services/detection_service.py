"""Detection telemetry service (Phase 20 §3).

If endpoint detection telemetry is intentionally shared, expose only the minimum
information needed by the control center. Do not upload raw malware files.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import time


class DetectionService:
    """In-memory minimal telemetry store (server must not receive raw files)."""

    def __init__(self):
        self._telemetry: List[Dict[str, Any]] = []

    def submit_telemetry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Enforce no raw files
        if any(k in payload for k in ("raw_file", "file_bytes", "pe_bytes", "raw_bytes", "file_content")):
            raise ValueError("raw malware files must not be uploaded")
        # Only keep minimum fields
        allowed = {"client_id", "detection_id", "timestamp", "sha256", "file_type", "model_version", "malware_probability", "risk_score", "verdict", "action"}
        filtered = {k: v for k, v in payload.items() if k in allowed}
        filtered["received_at"] = time.time()
        self._telemetry.append(filtered)
        # Keep last 1000
        if len(self._telemetry) > 1000:
            self._telemetry = self._telemetry[-1000:]
        return {"status": "received", "detection_id": filtered.get("detection_id")}

    def list_telemetry(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._telemetry[-limit:]

    def get_telemetry(self, detection_id: str) -> Optional[Dict[str, Any]]:
        for t in self._telemetry:
            if t.get("detection_id") == detection_id:
                return t
        return None
