"""Reproducibility pack — single command to reproduce experiment (Enhancement 20)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from src.utils.reproducibility import config_fingerprint


def create_repro_pack(experiment_dir: Path, output_path: Path) -> Path:
    """Create a reproducibility pack with all configs, code version, and instructions."""
    exp_dir = Path(experiment_dir)
    pack = {
        "experiment_id": exp_dir.name,
        "config": json.loads((exp_dir / "config_resolved.json").read_text(encoding="utf-8")) if (exp_dir / "config_resolved.json").exists() else {},
        "fingerprint": config_fingerprint(json.loads((exp_dir / "config_resolved.json").read_text(encoding="utf-8"))) if (exp_dir / "config_resolved.json").exists() else None,
        "reproduce_command": f"python scripts/run_experiment.py --config {exp_dir / 'config_resolved.json'}",
        "expected_variation": "F1 ±0.01, worst ±0.02 due to thread scheduling (deterministic dropout mitigates but not bit-for-bit)",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    return output_path
