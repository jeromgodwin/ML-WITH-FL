"""Drift service (Phase 20 §6).

Expose drift status, drift score, last drift event, adaptive retraining event, model version.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class DriftService:
    def __init__(self):
        self._status: str = "NO_DRIFT"
        self._score: Optional[float] = None
        self._last_event: Optional[Dict[str, Any]] = None
        self._retraining_events: List[Dict[str, Any]] = []
        self._model_version: Optional[str] = None

    def update_drift(self, status: str, score: Optional[float], model_version: Optional[str] = None) -> None:
        self._status = status
        self._score = score
        if score is not None and status in ("DRIFT_SUSPECTED", "DRIFT_DETECTED"):
            self._last_event = {"status": status, "score": score, "timestamp": time.time(), "model_version": model_version}
        if model_version:
            self._model_version = model_version

    def record_retraining(self, event: Dict[str, Any]) -> None:
        event["timestamp"] = time.time()
        self._retraining_events.append(event)
        if len(self._retraining_events) > 100:
            self._retraining_events = self._retraining_events[-100:]

    def get_status(self) -> Dict[str, Any]:
        return {"drift_status": self._status, "drift_score": self._score}

    def get_last_event(self) -> Optional[Dict[str, Any]]:
        return self._last_event

    def get_retraining_events(self) -> List[Dict[str, Any]]:
        return self._retraining_events

    def get_model_version(self) -> Optional[str]:
        return self._model_version
