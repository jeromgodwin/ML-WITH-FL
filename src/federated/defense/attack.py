"""Controlled malicious-client simulation (Phase 14).

Simulates intentionally abnormal model updates ONLY — no real malware, no
operational attack tooling. The purpose is to measure attack impact,
detection capability, and mitigation capability of the defenses.

Attack types:

- label_flip: the malicious client trains on locally flipped labels
  (data-level poisoning); its update is honest for its poisoned data.
- scaled_update: the malicious client trains honestly, then multiplies its
  parameter update by ``update_scale`` before returning it (abnormal
  update magnitude).
- replacement: the malicious client returns a large random parameter
  vector unrelated to local training (out-of-distribution update).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class AttackSpec:
    """Attack parameters for one experiment run."""

    enabled: bool = False
    attack_type: str = "none"
    n_malicious: int = 2
    update_scale: float = 20.0
    flip_frac: float = 1.0
    seed: int = 42

    @classmethod
    def from_config(cls, cfg) -> "AttackSpec":
        return cls(
            enabled=bool(getattr(cfg, "enabled", False)),
            attack_type=str(getattr(cfg, "attack_type", "none")),
            n_malicious=int(getattr(cfg, "n_malicious", 2)),
            update_scale=float(getattr(cfg, "update_scale", 20.0)),
            flip_frac=float(getattr(cfg, "flip_frac", 1.0)),
            seed=int(getattr(cfg, "seed", 42)),
        )

    @property
    def is_malicious(self) -> bool:
        return self.enabled and self.attack_type != "none"

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "attack_type": self.attack_type,
            "n_malicious": self.n_malicious,
            "update_scale": self.update_scale,
            "flip_frac": self.flip_frac,
            "seed": self.seed,
        }


def is_malicious_cid(cid: int, spec: AttackSpec) -> bool:
    """The first ``n_malicious`` clients of the partition are malicious."""
    if not spec.is_malicious:
        return False
    return cid < spec.n_malicious


def flip_labels(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Flip a fraction of binary labels (0<->1) deterministically."""
    y = np.asarray(y).copy()
    n_flip = max(1, int(round(len(y) * frac)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), n_flip, replace=False)
    y[idx] = 1 - y[idx]
    return y


def apply_attack(
    returned_params: List[np.ndarray],
    received_params: List[np.ndarray],
    cid: int,
    spec: AttackSpec,
) -> List[np.ndarray]:
    """Transform an honest client's returned parameters into the attack.

    For ``scaled_update`` the returned update (returned - received) is
    scaled by ``update_scale`` and re-added to the received global.
    For ``replacement`` a seeded large random vector replaces the update
    entirely. ``label_flip`` alters only the training data, not the
    returned parameters, and is handled in the client's fit loop.
    """
    if not is_malicious_cid(cid, spec):
        return returned_params
    if spec.attack_type == "label_flip":
        return returned_params
    rng = np.random.default_rng(spec.seed + cid)
    if spec.attack_type == "replacement":
        scale = 10.0 * (spec.update_scale or 20.0)
        return [rng.standard_normal(p.shape).astype(np.float32) * scale
                for p in returned_params]
    # scaled_update
    g = [np.asarray(p, dtype=np.float32) for p in received_params]
    new = [np.asarray(p, dtype=np.float32) for p in returned_params]
    return [gg + (nw - gg) * spec.update_scale for gg, nw in zip(g, new)]


def attack_report(spec: AttackSpec, malicious_cids: List[int]) -> dict:
    """Human-readable description of the simulated attack (for reports)."""
    return {
        "attack_type": spec.attack_type if spec.enabled else "none",
        "n_malicious": len(malicious_cids) if spec.enabled else 0,
        "malicious_cids": malicious_cids if spec.enabled else [],
        "update_scale": spec.update_scale,
        "flip_frac": spec.flip_frac,
        "note": "synthetic abnormal updates only — no real malware",
    }
