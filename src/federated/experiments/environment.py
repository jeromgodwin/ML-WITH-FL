"""Environment and reproducibility metadata for Phase 15.

Collects software versions, hardware hints, git state, and deterministic
reproducibility records without breaking existing endpoint/FL code.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fedshield.config import ExperimentConfig


def collect_environment_metadata() -> Dict[str, Any]:
    """Capture environment metadata where practical."""
    meta: Dict[str, Any] = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "executable": sys.executable,
    }
    # Optional deps
    for pkg in ("torch", "numpy", "pandas", "sklearn", "flwr", "yaml", "psutil", "tqdm"):
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", None)
            if ver is None and pkg == "sklearn":
                import sklearn as _sk

                ver = _sk.__version__
            if ver is None and pkg == "yaml":
                import yaml as _y

                ver = getattr(_y, "__version__", "unknown")
            meta[f"{pkg}_version"] = str(ver) if ver else "unknown"
        except Exception:
            meta[f"{pkg}_version"] = "unknown"

    # pip freeze hash (best-effort, for reproducibility)
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, timeout=10)
        meta["pip_freeze_sha256"] = hashlib.sha256(freeze.encode()).hexdigest()[:12]
    except Exception:
        meta["pip_freeze_sha256"] = "unknown"

    # git state (best-effort)
    try:
        root = _find_git_root()
        if root:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(root), text=True, timeout=5
            ).strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(root), text=True, timeout=5
            ).strip()
            dirty = subprocess.call(
                ["git", "diff", "--quiet"], cwd=str(root), timeout=5
            ) != 0
            meta["git_commit"] = sha
            meta["git_branch"] = branch
            meta["git_dirty"] = bool(dirty)
    except Exception:
        meta["git_commit"] = "unknown"
        meta["git_branch"] = "unknown"
        meta["git_dirty"] = "unknown"

    return meta


def collect_reproducibility_record(
    cfg: ExperimentConfig,
    partition_dir: Optional[Path] = None,
    vectorized_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build reproducibility record for a resolved config."""
    rec: Dict[str, Any] = {
        "seed": cfg.seed,
        "partition_seed": cfg.partition.seed,
        "train_seed": cfg.train.seed,
        "attack_seed": cfg.attack.seed,
        "dataset_version": cfg.data.ember_version,
        "dataset_data_dir": cfg.data.data_dir,
        "model_config": cfg.model.__dict__ if hasattr(cfg.model, "__dict__") else dict(cfg.model.__dict__),
        "preprocessing_version": _preprocessing_version(vectorized_dir),
        "algorithm_config": {
            "algorithm": cfg.fl.algorithm,
            "proximal_mu": cfg.fl.proximal_mu,
            "personalized_probe_samples": cfg.fl.personalized_probe_samples,
            "personalized_probe_epochs": cfg.fl.personalized_probe_epochs,
            "personalized_probe_learning_rate": cfg.fl.personalized_probe_learning_rate,
            "personalized_head_epochs": cfg.fl.personalized_head_epochs,
            "personalized_head_learning_rate": cfg.fl.personalized_head_learning_rate,
        },
    }
    # Partition reference
    if partition_dir and Path(partition_dir).exists():
        try:
            # hash manifest if present
            mf = Path(partition_dir) / "manifest.json"
            if mf.exists():
                rec["partition_manifest_sha256"] = _file_sha256(mf)
            # record partition config if available
            from src.federated.data.partition import load_partition

            pc, *_ = load_partition(Path(partition_dir))
            rec["partition_config"] = pc.to_dict() if hasattr(pc, "to_dict") else str(pc)
        except Exception as e:
            rec["partition_error"] = str(e)
    # dataset reference hash (vectorized dir listing)
    if vectorized_dir and Path(vectorized_dir).exists():
        rec["vectorized_dir"] = str(vectorized_dir)
        try:
            files = sorted([p.name for p in Path(vectorized_dir).glob("*.npy")])
            rec["vectorized_files"] = files
        except Exception:
            pass
    # config fingerprint
    try:
        from src.utils.reproducibility import config_fingerprint

        rec["config_fingerprint"] = config_fingerprint(cfg.to_dict())
    except Exception:
        pass
    # software versions snapshot (subset)
    rec["software"] = {
        k: v
        for k, v in collect_environment_metadata().items()
        if k.endswith("_version") or k.startswith("git_")
    }
    return rec


def _preprocessing_version(vectorized_dir: Optional[Path]) -> Optional[str]:
    if vectorized_dir is None:
        return None
    # Look for scaler manifest or version file
    candidates = [
        Path(vectorized_dir).parent / "artifacts" / "manifest.json",
        Path(vectorized_dir) / "preprocessing.json",
        Path("data/ember_2018_2/artifacts/manifest.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return _file_sha256(p)[:12]
            except Exception:
                continue
    return None


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_git_root(start: Optional[Path] = None) -> Optional[Path]:
    cur = Path(start) if start else Path(__file__).resolve().parent
    for _ in range(6):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # fallback to repo root via pyproject
    cur = Path(__file__).resolve()
    while cur != cur.parent:
        if (cur / "pyproject.toml").exists():
            return cur
        cur = cur.parent
    return None
