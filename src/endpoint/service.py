"""Dashboard service interface (Phase 8).

A clean Python API that the future React dashboard will consume (via the
Phase-10 backend). It intentionally exposes only JSON-serializable dicts with
user-facing fields and no internal machinery. The React UI is NOT built here.

Methods:
    get_status()                        monitor/engine status snapshot
    get_detections(limit)               latest detections
    search_history(...)                 filtered history (verdict/timestamp/model)
    get_quarantined()                   quarantined files
    get_models()                        registered models (from ModelRegistry)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fedshield.config import QuarantineConfig
from fedshield.logging_setup import get_logger
from src.endpoint.history import HistoryStore
from src.endpoint.quarantine import QuarantineManager
from src.interfaces import DetectionRecord

logger = get_logger(__name__)


class DashboardService:
    def __init__(
        self,
        history: HistoryStore,
        quarantine: Optional[QuarantineManager] = None,
        monitor: Optional[Any] = None,          # FileMonitor with .status()
        registry: Optional[Any] = None,         # ModelRegistry
        quarantine_dir: Optional[Path] = None,  # fallback if manager not given
    ):
        self.history = history
        self.quarantine = quarantine or (
            QuarantineManager(QuarantineConfig(quarantine_dir=str(quarantine_dir or "quarantine")))
            if quarantine_dir is not None else None)
        self.monitor = monitor
        self.registry = registry

    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {"service": "fedshield-endpoint"}
        if self.monitor is not None:
            try:
                s = self.monitor.status()
                status["monitor"] = {k: s[k] for k in ("running", "files_seen",
                                                       "files_analyzed", "errors")
                                     if k in s}
            except Exception as exc:
                status["monitor"] = {"running": False, "error": str(exc)}
        else:
            status["monitor"] = {"running": False, "reason": "monitor not attached"}
        status["history_entries"] = self.history.count()
        status["quarantined_count"] = len(self.quarantine.list()) if self.quarantine else 0
        return status

    def get_detections(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.history.latest(limit=limit)

    def search_history(
        self,
        verdict: Optional[str] = None,
        action: Optional[str] = None,
        model_version: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        min_risk_score: Optional[int] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        return self.history.search(
            verdict=verdict, action=action, model_version=model_version,
            since=since, until=until, min_risk_score=min_risk_score, limit=limit)

    def get_quarantined(self) -> List[Dict[str, Any]]:
        if self.quarantine is None:
            return []
        out = []
        for qr in self.quarantine.list():
            d = qr.to_dict()
            d["detection"] = self._summarize_detection(d["detection"])
            out.append(d)
        return out

    def get_models(self) -> List[Dict[str, Any]]:
        if self.registry is None:
            return []
        try:
            return [m.to_dict() for m in self.registry.list_models()]
        except Exception as exc:
            logger.error("cannot list models: %s", exc)
            return []

    # ------------------------------------------------------------------
    @staticmethod
    def _summarize_detection(d: Dict[str, Any]) -> Dict[str, Any]:
        """Strip internal fields before exposing detection data to a UI."""
        return {k: d[k] for k in (
            "detection_id", "timestamp", "filename", "filepath", "sha256",
            "file_type", "model_version", "malware_probability",
            "benign_probability", "risk_score", "risk_level", "verdict",
            "action", "analysis_duration_ms",
        ) if k in d}