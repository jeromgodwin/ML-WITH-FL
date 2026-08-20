"""Configurable model-update norm clipping (Phase 14).

Clips each client's parameter UPDATE (delta between the received global
parameters and the returned local parameters) to a maximum L2 norm.
Records the norm before clipping, the threshold, and the norm after.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ClipRecord:
    """Per-update clipping measurement."""

    norm_before: float
    threshold: Optional[float]
    norm_after: float
    clipped: bool
    scale: float

    def to_dict(self) -> dict:
        return {
            "norm_before": round(float(self.norm_before), 6),
            "threshold": None if self.threshold is None else round(float(self.threshold), 6),
            "norm_after": round(float(self.norm_after), 6),
            "clipped": bool(self.clipped),
            "scale": round(float(self.scale), 6),
        }


class UpdateClipper:
    """Clips update vectors to a maximum L2 norm.

    ``max_norm`` is the clipping threshold; None disables clipping (all
    updates pass through unchanged, with records still produced).
    """

    def __init__(self, max_norm: Optional[float] = None):
        if max_norm is not None and max_norm <= 0:
            raise ValueError(f"clip threshold must be > 0, got {max_norm}")
        self.max_norm = max_norm
        self.records: List[ClipRecord] = []

    def clip(self, update: np.ndarray) -> np.ndarray:
        """Clip one update vector; record before/threshold/after."""
        norm = float(np.linalg.norm(update))
        threshold = self.max_norm
        if threshold is None or norm <= threshold:
            clipped_update = update
            scale = 1.0
            clipped = False
        else:
            scale = threshold / norm
            clipped_update = update * scale
            clipped = True
        rec = ClipRecord(
            norm_before=norm,
            threshold=threshold,
            norm_after=float(np.linalg.norm(clipped_update)),
            clipped=clipped,
            scale=scale,
        )
        self.records.append(rec)
        return clipped_update

    def clear(self) -> None:
        self.records = []

    def summary(self) -> dict:
        if not self.records:
            return {"n_updates": 0}
        norms_before = [r.norm_before for r in self.records]
        n_clipped = sum(1 for r in self.records if r.clipped)
        return {
            "n_updates": len(self.records),
            "n_clipped": n_clipped,
            "max_norm_before": round(max(norms_before), 6),
            "mean_norm_before": round(sum(norms_before) / len(norms_before), 6),
            "clip_threshold": self.max_norm,
        }
