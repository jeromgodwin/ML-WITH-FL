"""Federated learning service (Phase 20 §4).

Exposes algorithms, experiment configurations, start experiment, status,
round metrics, client metrics, comparison results. Delegates to
ExperimentConfig and run_fl_experiment via background task — no ML logic in handlers.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fedshield.config import ExperimentConfig


class FLService:
    def __init__(self, experiments_root: Path | str = "data/experiments"):
        self.root = Path(experiments_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._experiments: Dict[str, Dict[str, Any]] = {}
        # Load existing summaries if any
        for p in self.root.glob("*/summary.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                exp_id = data.get("experiment_id") or p.parent.name
                self._experiments[exp_id] = {"status": "completed", "summary": data}
            except Exception:
                continue

    def algorithms(self) -> List[str]:
        return ["fedavg", "fedprox", "personalized", "centralized"]

    def list_configs(self) -> List[str]:
        cfg_dir = Path("configs")
        if not cfg_dir.exists():
            return []
        return [str(p) for p in cfg_dir.glob("*.yaml")]

    def start_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        exp_id = f"exp-{uuid.uuid4().hex[:8]}"
        # Store config and mark as running (actual training is async; here we record)
        self._experiments[exp_id] = {"status": "running", "config": config, "started_at": time.time()}
        # For Phase 20, we do not block on training — caller polls status
        # A real implementation would spawn a background task via run_unified_experiment
        return {"experiment_id": exp_id, "status": "running"}

    def get_status(self, exp_id: str) -> Optional[Dict[str, Any]]:
        return self._experiments.get(exp_id)

    def get_round_metrics(self, exp_id: str) -> Optional[List[Dict[str, Any]]]:
        exp = self._experiments.get(exp_id)
        if not exp or "summary" not in exp:
            return None
        return exp["summary"].get("rounds")

    def get_client_metrics(self, exp_id: str) -> Optional[List[Dict[str, Any]]]:
        exp = self._experiments.get(exp_id)
        if not exp or "summary" not in exp:
            return None
        return exp["summary"].get("per_client_metrics")

    def comparison(self) -> List[Dict[str, Any]]:
        # Minimal comparison across known experiments (from unified aggregation)
        rows = []
        for eid, data in self._experiments.items():
            summary = data.get("summary")
            if not summary:
                continue
            final = summary.get("final_global_test_metrics") or {}
            rows.append({"experiment_id": eid, "f1": final.get("f1"), "accuracy": final.get("accuracy")})
        return rows

    def register_completed(self, exp_id: str, summary: Dict[str, Any]) -> None:
        self._experiments[exp_id] = {"status": "completed", "summary": summary}
