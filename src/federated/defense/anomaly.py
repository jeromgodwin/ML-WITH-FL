"""Per-update anomaly detection (Phase 14).

Computes an anomaly score per client update from three signals:

- update magnitude (L2 norm of the update vector)
- deviation from peer updates (cosine distance from the mean update
  direction)
- distance from the reference/expected update (L2 distance from the
  rolling median of recent accepted global update norms)

Scores are standardized against the robust scale (MAD) of the round's
peer scores, then classified:

    NORMAL          score <  suspect_mult
    SUSPICIOUS      suspect_mult <= score < detect_mult
    HIGHLY_ANOMALOUS  score >= detect_mult

This is a measurement tool, not a claim of complete poisoning detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class AnomalyClass(str, Enum):
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    HIGHLY_ANOMALOUS = "HIGHLY_ANOMALOUS"


@dataclass
class AnomalyRecord:
    """Per-update anomaly measurement."""

    cid: str
    update_norm: float
    peer_deviation: float
    reference_distance: float
    score: float
    classification: AnomalyClass

    def to_dict(self) -> dict:
        return {
            "cid": self.cid,
            "update_norm": round(float(self.update_norm), 6),
            "peer_deviation": round(float(self.peer_deviation), 6),
            "reference_distance": round(float(self.reference_distance), 6),
            "score": round(float(self.score), 6),
            "classification": self.classification.value,
        }


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation (robust scale)."""
    median = np.median(x)
    return float(np.median(np.abs(x - median)))


class UpdateAnomalyDetector:
    """Scores a batch of client updates and classifies each one.

    ``reference_norm`` is the expected update magnitude (e.g. the rolling
    median of recently accepted global update norms); None disables the
    reference-distance signal.
    """

    def __init__(
        self,
        suspect_mult: float = 3.0,
        detect_mult: float = 6.0,
        reference_norm: Optional[float] = None,
    ):
        if detect_mult <= suspect_mult:
            raise ValueError("detect_mult must be > suspect_mult")
        self.suspect_mult = suspect_mult
        self.detect_mult = detect_mult
        self.reference_norm = reference_norm
        self.records: List[AnomalyRecord] = []
        self.rounds: List[List[AnomalyRecord]] = []

    def _signal_norms(
        self,
        updates: List[Tuple[str, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        norms = np.array([float(np.linalg.norm(u)) for _, u in updates], dtype=float)
        # Deviation from the mean peer direction (cosine distance).
        if len(updates) > 1:
            directions = np.array([u / n if n > 0 else u for (_, u), n in
                                   zip(updates, norms)], dtype=float)
            mean_dir = directions.mean(axis=0)
            mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-12)
            cos = np.clip(directions @ mean_dir, -1.0, 1.0)
            peer_dev = 1.0 - cos
        else:
            peer_dev = np.zeros(len(updates))
        # Distance from the reference/expected update magnitude.
        if self.reference_norm is not None and self.reference_norm > 0:
            ref_dist = np.abs(norms - self.reference_norm) / self.reference_norm
        else:
            ref_dist = np.zeros(len(updates))
        return norms, peer_dev, ref_dist

    def score_and_classify(
        self,
        updates: List[Tuple[str, np.ndarray]],
        reference_norm: Optional[float] = None,
    ) -> List[AnomalyRecord]:
        """Score one round of updates; returns per-client records."""
        if reference_norm is not None:
            self.reference_norm = reference_norm
        norms, peer_dev, ref_dist = self._signal_norms(updates)
        # Standardize the composite signal against the robust peer scale.
        raw = norms + peer_dev + ref_dist
        scale = _mad(raw) if len(raw) > 1 else 0.0
        if scale > 0:
            scores = (raw - np.median(raw)) / scale
        else:
            scores = np.zeros(len(raw))

        records: List[AnomalyRecord] = []
        for (cid, _), n, pd, rd, s in zip(updates, norms, peer_dev, ref_dist, scores):
            if s >= self.detect_mult:
                cls = AnomalyClass.HIGHLY_ANOMALOUS
            elif s >= self.suspect_mult:
                cls = AnomalyClass.SUSPICIOUS
            else:
                cls = AnomalyClass.NORMAL
            records.append(AnomalyRecord(cid=cid, update_norm=n,
                                         peer_deviation=pd,
                                         reference_distance=rd,
                                         score=s, classification=cls))
        self.records.extend(records)
        self.rounds.append(records)
        return records

    def flagged_cids(self) -> List[str]:
        """Client ids ever flagged SUSPICIOUS or worse."""
        return [r.cid for r in self.records
                if r.classification != AnomalyClass.NORMAL]

    def summary(self) -> dict:
        if not self.records:
            return {"n_updates": 0}
        n_susp = sum(1 for r in self.records
                     if r.classification == AnomalyClass.SUSPICIOUS)
        n_high = sum(1 for r in self.records
                     if r.classification == AnomalyClass.HIGHLY_ANOMALOUS)
        return {
            "n_updates": len(self.records),
            "n_normal": len(self.records) - n_susp - n_high,
            "n_suspicious": n_susp,
            "n_highly_anomalous": n_high,
            "flagged_cids": sorted(set(self.flagged_cids())),
        }
