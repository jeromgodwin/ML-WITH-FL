"""Monitor service — exposes file arrival feed + scan process + vulnerability (Phase 21).

Aggregates:
- FileMonitor status (data/monitor/status.json written by scripts/run_monitor.py)
- Recent all-files feed (every file that reached watched dirs, PE + non-PE)
- Recent detections with vulnerability details (data/monitor/detections.jsonl + endpoint history)

No raw files are ever returned — only metadata, hashes, probabilities, risk, explanation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class MonitorService:
    def __init__(
        self,
        status_path: Path | str = "data/monitor/status.json",
        detections_path: Path | str = "data/monitor/detections.jsonl",
        history_db: Path | str = "history/detections.db",
    ):
        self.status_path = Path(status_path)
        self.detections_path = Path(detections_path)
        self.history_db = Path(history_db)

    def get_status(self) -> Dict[str, Any]:
        """Return FileMonitor snapshot + aggregated vulnerability feed."""
        status = {}
        # Try to read FileMonitor status file (written every second by run_monitor)
        if self.status_path.exists():
            try:
                status = json.loads(self.status_path.read_text(encoding="utf-8"))
            except Exception:
                status = {}
        # Fallback: try endpoint config for watched dirs
        if not status:
            try:
                from fedshield.config import ExperimentConfig
                cfg = ExperimentConfig.from_yaml("configs/default.yaml")
                status = {
                    "running": False,
                    "watched_directories": list(cfg.endpoint.monitor.watched_directories),
                    "recursive": cfg.endpoint.monitor.recursive,
                    "poll_interval": cfg.endpoint.monitor.poll_interval,
                    "all_files_seen": 0,
                    "files_seen": 0,
                    "files_analyzed": 0,
                    "recent_all_files": [],
                    "recent_events": [],
                    "note": "monitor not running — start: python scripts/run_monitor.py",
                }
            except Exception as e:
                status = {"running": False, "error": str(e), "watched_directories": []}

        # Enrich with recent detections vulnerability summary
        detections = self.get_recent_detections(limit=20)
        if detections:
            # attach to status for convenience
            status["recent_detections"] = detections
            status["detections_count"] = len(detections)
        else:
            status.setdefault("recent_detections", [])
        return status

    def get_recent_files(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Every file arrival (PE + non-PE) with scan_status + vulnerability."""
        status = self.get_status()
        # Prefer recent_all_files from FileMonitor (has vulnerability for all)
        all_files = status.get("recent_all_files") or []
        if all_files:
            return all_files[-limit:][::-1] if len(all_files) > limit else list(reversed(all_files))
        # Fallback: derive from detections.jsonl if monitor not running
        return self.get_recent_detections(limit=limit)

    def get_recent_detections(self, limit: int = 20) -> List[Dict[str, Any]]:
        """PE detections with full vulnerability details (malware scan process)."""
        out: List[Dict[str, Any]] = []
        # 1) Try detections.jsonl (run_monitor's file, has full scan_result + vulnerability)
        if self.detections_path.exists():
            try:
                lines = self.detections_path.read_text(encoding="utf-8").strip().splitlines()
                for line in lines[-limit:]:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
            except Exception:
                pass
        # 2) Try HistoryStore SQLite (endpoint local history)
        if not out and self.history_db.exists():
            try:
                from src.endpoint.history import HistoryStore
                hs = HistoryStore(db_path=self.history_db)
                rows = hs.latest(limit=limit)
                # enrich with vulnerability hint
                for r in rows:
                    r["vulnerability"] = {
                        "risk_score": r.get("risk_score"),
                        "risk_level": r.get("risk_level"),
                        "verdict": r.get("verdict"),
                        "action": r.get("action"),
                        "malware_probability": r.get("malware_probability"),
                    }
                    r["scan_process"] = ["stable", "sha256", "feature_extraction", "inference", "risk_engine", "verdict"]
                out = rows
            except Exception:
                pass
        # Most recent first
        out = list(reversed(out)) if out and "timestamp" in out[0] else out
        return out[:limit]

    def get_file_by_sha(self, sha256: str) -> Optional[Dict[str, Any]]:
        for d in self.get_recent_detections(limit=1000):
            if d.get("sha256") == sha256:
                return d
        for f in self.get_recent_files(limit=1000):
            if f.get("sha256") == sha256:
                return f
        return None
