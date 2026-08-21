"""Experiment matrix definitions for Phase 15.

Supports:
- Centralized
- FedAvg: IID / Mild / Moderate / Severe
- FedProx: IID / Mild / Moderate / Severe
- Personalized FL: IID / Mild / Moderate / Severe
- Controlled: resource awareness, concept drift, poisoning defenses, privacy
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from fedshield.config import ExperimentConfig

NONIID_LEVELS = ["iid", "mild", "moderate", "severe"]
FL_ALGORITHMS = ["fedavg", "fedprox", "personalized"]


def _base_cfg(seed: int = 42) -> ExperimentConfig:
    return ExperimentConfig.from_dict({"seed": seed, "partition": {"seed": seed}})


def centralized_entry(seed: int = 42, **overrides: Any) -> Dict[str, Any]:
    cfg = _base_cfg(seed).with_overrides(**overrides)
    cfg.fl.algorithm = "centralized"
    # centralized uses train.epochs, not fl rounds
    return {
        "name": "centralized",
        "label": "centralized",
        "kind": "centralized",
        "config": cfg,
        "meta": {"algorithm": "centralized", "strategy": "centralized"},
    }


def fl_entry(
    algorithm: str,
    strategy: str,
    seed: int = 42,
    **overrides: Any,
) -> Dict[str, Any]:
    cfg = _base_cfg(seed)
    cfg.partition.strategy = strategy
    cfg.fl.algorithm = algorithm
    if algorithm == "fedprox" and cfg.fl.proximal_mu == 0.0:
        cfg.fl.proximal_mu = 0.1
    # apply overrides
    if overrides:
        cfg = cfg.with_overrides(**overrides)
    label = f"{algorithm}-{strategy}"
    return {
        "name": label,
        "label": label,
        "kind": "federated",
        "config": cfg,
        "meta": {"algorithm": algorithm, "strategy": strategy},
    }


def full_matrix(seed: int = 42, include_centralized: bool = True) -> List[Dict[str, Any]]:
    """Generate the full Phase-15 matrix (centralized + 12 FL cells)."""
    entries: List[Dict[str, Any]] = []
    if include_centralized:
        entries.append(centralized_entry(seed=seed))
    for algo in FL_ALGORITHMS:
        for strat in NONIID_LEVELS:
            entries.append(fl_entry(algo, strat, seed=seed))
    return entries


def controlled_entries(seed: int = 42) -> List[Dict[str, Any]]:
    """Controlled experiments: resource, drift, poisoning, privacy."""
    entries: List[Dict[str, Any]] = []
    # resource awareness: resource-aware vs normal (iid, fedavg)
    for enabled in (False, True):
        cfg = _base_cfg(seed)
        cfg.fl.algorithm = "fedavg"
        cfg.partition.strategy = "iid"
        cfg.endpoint.resource.enabled = enabled
        if enabled:
            cfg.endpoint.resource.max_cpu_percent = 70.0
            cfg.endpoint.resource.check_interval_sec = 1.0
        label = f"resource-{'aware' if enabled else 'normal'}"
        entries.append(
            {"name": label, "label": label, "kind": "controlled_resource", "config": cfg, "meta": {"resource_enabled": enabled}}
        )
    # concept drift: drift enabled vs disabled
    for enabled in (False, True):
        cfg = _base_cfg(seed)
        cfg.fl.algorithm = "fedavg"
        cfg.partition.strategy = "iid"
        cfg.endpoint.drift.enabled = enabled
        label = f"drift-{'enabled' if enabled else 'disabled'}"
        entries.append(
            {"name": label, "label": label, "kind": "controlled_drift", "config": cfg, "meta": {"drift_enabled": enabled}}
        )
    # poisoning defenses: each defense mode on same attack
    for mode in ("none", "clipping", "anomaly", "validation", "robust_median", "robust_trimmed"):
        cfg = _base_cfg(seed)
        cfg.fl.algorithm = "fedavg"
        cfg.partition.strategy = "iid"
        cfg.attack.enabled = True
        cfg.attack.attack_type = "scaled_update"
        cfg.attack.n_malicious = 2
        cfg.defense.mode = mode
        if mode == "clipping":
            cfg.defense.clip_norm = 0.05
        label = f"poisoning-{mode}"
        entries.append(
            {"name": label, "label": label, "kind": "controlled_poisoning", "config": cfg, "meta": {"defense_mode": mode}}
        )
    # privacy: enabled vs disabled
    for enabled in (False, True):
        cfg = _base_cfg(seed)
        cfg.fl.algorithm = "fedavg"
        cfg.partition.strategy = "iid"
        cfg.privacy.enabled = enabled
        if enabled:
            cfg.privacy.noise_multiplier = 1.0
            cfg.privacy.max_grad_norm = 1.0
        label = f"privacy-{'enabled' if enabled else 'disabled'}"
        entries.append(
            {"name": label, "label": label, "kind": "controlled_privacy", "config": cfg, "meta": {"privacy_enabled": enabled}}
        )
    return entries


def matrix_from_config(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build matrix from a user YAML/JSON dict.

    Expected keys:
      matrix: list of dicts with keys:
        name (optional), algorithm, strategy, overrides (optional dict of dotted overrides)
      include_centralized (bool)
      seed (int)
    If 'matrix' is absent, returns full_matrix().
    """
    if "matrix" not in raw:
        seed = int(raw.get("seed", 42))
        include = bool(raw.get("include_centralized", True))
        entries = full_matrix(seed=seed, include_centralized=include)
        # optionally extend with controlled if requested
        if raw.get("include_controlled"):
            entries.extend(controlled_entries(seed=seed))
        return entries
    seed = int(raw.get("seed", 42))
    entries: List[Dict[str, Any]] = []
    for item in raw["matrix"]:
        if not isinstance(item, dict):
            continue
        algo = item.get("algorithm", "fedavg")
        strat = item.get("strategy", "iid")
        name = item.get("name")
        overrides = item.get("overrides") or {}
        kind = item.get("kind", "federated" if algo != "centralized" else "centralized")
        if algo == "centralized":
            e = centralized_entry(seed=seed, **{k.replace(".", "."): v for k, v in overrides.items()})
            if name:
                e["name"] = name
                e["label"] = name
        else:
            e = fl_entry(algo, strat, seed=seed, **{k: v for k, v in overrides.items()})
            if name:
                e["name"] = name
                e["label"] = name
        if kind:
            e["kind"] = kind
        entries.append(e)
    return entries
