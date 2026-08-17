"""Reproducibility utilities: seeds, experiment IDs, config snapshots, paths.

No machine-specific absolute paths are hardcoded here; paths are resolved
relative to a configurable project root.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from fedshield.logging_setup import get_logger

logger = get_logger(__name__)


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch not installed in this environment
        pass


def generate_experiment_id(name: Optional[str] = None, seed: Optional[int] = None) -> str:
    """Generate a unique, human-readable experiment ID.

    Format: <name>-<yyyymmdd-hhmmss>-<8-hex-uuid>.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    prefix = "exp"
    if name:
        prefix = "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-") or "exp"
    exp_id = f"{prefix}-{stamp}-{suffix}"
    if seed is not None:
        exp_id += f"-s{seed}"
    return exp_id


def config_fingerprint(config_dict: dict[str, Any]) -> str:
    """Deterministic SHA-256 of a config snapshot (for change detection)."""
    blob = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def save_config_snapshot(config_dict: dict[str, Any], target_dir: str | Path) -> Path:
    """Persist a config snapshot to ``target_dir/config_snapshot.json``.

    Returns the path written.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / "config_snapshot.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, sort_keys=True)
    return out


def load_config_snapshot(path: str | Path) -> dict[str, Any]:
    """Load a previously saved config snapshot."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_project_path(path: str | Path, project_root: Optional[str | Path] = None) -> Path:
    """Resolve a possibly-relative, possibly-``~``-prefixed path.

    ``project_root`` defaults to the repo root (directory containing ``pyproject.toml``).
    """
    p = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if p.is_absolute():
        return p
    root = Path(project_root) if project_root else _default_project_root()
    return (root / p).resolve()


def _default_project_root() -> Path:
    """Locate the repo root by walking up to the directory with pyproject.toml."""
    current = Path(__file__).resolve().parent.parent.parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


def ensure_dir(path: str | Path, project_root: Optional[str | Path] = None) -> Path:
    """Resolve a path and create it (parents included); returns the dir."""
    resolved = resolve_project_path(path, project_root)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
