"""Result storage with unique experiment IDs and never-overwrite guarantee."""

from __future__ import annotations

import json
import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fedshield.config import ExperimentConfig
from src.utils.reproducibility import generate_experiment_id, config_fingerprint

DEFAULT_EXPERIMENTS_ROOT = Path("data/experiments")


class ExperimentStorage:
    """Handles on-disk layout for one unified experiment."""

    def __init__(self, root: Path | str, experiment_id: str):
        self.root = Path(root)
        self.experiment_id = experiment_id
        self.dir = self.root / experiment_id
        try:
            self.dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise FileExistsError(
                f"experiment directory already exists (never overwrite): {self.dir}"
            ) from None
        # subdirs
        (self.dir / "metrics").mkdir(parents=True, exist_ok=True)
        (self.dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.dir / "model").mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        cfg: ExperimentConfig,
        root: Path | str = DEFAULT_EXPERIMENTS_ROOT,
        experiment_id: Optional[str] = None,
    ) -> "ExperimentStorage":
        """Create a new storage with a unique ID."""
        if experiment_id is None:
            experiment_id = generate_experiment_id(name=cfg.name, seed=cfg.seed)
        # Ensure uniqueness even if timestamp collides (add suffix if exists)
        base = experiment_id
        counter = 0
        while (Path(root) / experiment_id).exists():
            counter += 1
            experiment_id = f"{base}-{counter}"
        return cls(root, experiment_id)

    # -- basic writers --
    def write_json(self, rel: str | Path, data: Any) -> Path:
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            raise FileExistsError(f"refusing to overwrite: {p}")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, default=str)
        return p

    def write_text(self, rel: str | Path, text: str) -> Path:
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            raise FileExistsError(f"refusing to overwrite: {p}")
        p.write_text(text, encoding="utf-8")
        return p

    def save_config(self, cfg: ExperimentConfig, raw_source: Optional[Dict[str, Any]] = None) -> None:
        # Resolved config
        self.write_json("config_resolved.json", cfg.to_dict())
        # Raw source if provided
        if raw_source is not None:
            self.write_json("config_input.json", raw_source)
        # Also write YAML copy for convenience
        try:
            import yaml

            yaml_path = self.dir / "config_input.yaml"
            if not yaml_path.exists() and raw_source is not None:
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(raw_source, f, sort_keys=False)
        except Exception:
            pass
        # fingerprint
        fp = config_fingerprint(cfg.to_dict())
        self.write_json("config_fingerprint.json", {"fingerprint": fp})

    def save_environment(self, env_meta: Dict[str, Any]) -> None:
        self.write_json("environment.json", env_meta)

    def save_reproducibility(self, rec: Dict[str, Any]) -> None:
        self.write_json("reproducibility.json", rec)

    def save_metrics(self, results: Dict[str, Any]) -> None:
        """Persist the unified result dict into structured files.

        Expected keys from engine: experiment, rounds, per_client_metrics,
        final_metrics, training_time, communication, resource, drift,
        security, privacy, model_metadata, logs.
        """
        # summary (full results)
        self.write_json("metrics/summary.json", results)
        # per-round
        rounds = results.get("rounds") or results.get("per_round_metrics") or []
        if rounds:
            with open(self.dir / "metrics/rounds.jsonl", "w", encoding="utf-8") as f:
                for r in rounds:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
            self.write_json("metrics/rounds.json", rounds)
        # per-client
        per_client = results.get("per_client_metrics") or results.get("per_client")
        if per_client is not None:
            self.write_json("metrics/per_client.json", per_client)
        # final metrics
        final = results.get("final_metrics") or results.get("final_global_test_metrics")
        if final is not None:
            self.write_json("metrics/final.json", final if isinstance(final, dict) else {"value": final})
        # training time
        if "training_time_s" in results:
            self.write_json("metrics/training_time.json", {"training_time_s": results["training_time_s"]})
        # communication
        if "communication" in results:
            self.write_json("metrics/communication.json", results["communication"])
        # resource
        if "resource" in results:
            self.write_json("metrics/resource.json", results["resource"])
        # drift
        if "drift" in results:
            self.write_json("metrics/drift.json", results["drift"])
        # security (attack + defense)
        if "security" in results:
            self.write_json("metrics/security.json", results["security"])
        elif "attack" in results or "defense" in results:
            sec = {}
            if "attack" in results:
                sec["attack"] = results["attack"]
            if "defense" in results:
                sec["defense"] = results["defense"]
            self.write_json("metrics/security.json", sec)
        # privacy
        if "privacy" in results:
            self.write_json("metrics/privacy.json", results["privacy"])
        # model metadata
        if "model_metadata" in results or "model" in results:
            self.write_json("model/metadata.json", results.get("model_metadata") or results.get("model") or {})
        # logs
        logs = results.get("logs")
        if logs is not None:
            self.write_json("logs/run.json", logs if isinstance(logs, dict) else {"logs": logs})

    def metadata(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dir": str(self.dir),
        }
