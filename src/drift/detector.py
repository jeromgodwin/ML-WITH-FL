"""Population Stability Index (PSI) drift detector (Phase 13).

Compares the distribution of feature values between a reference population
(historical training data) and a current population (new observations).
Returns a drift status: NO_DRIFT, DRIFT_SUSPECTED, DRIFT_DETECTED.

PSI is computed per feature and aggregated (mean PSI). PSI ~0 means identical
distributions; >0.1 suggests moderate shift; >0.2 strong shift.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from fedshield.config import DriftConfig
from fedshield.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DriftResult:
    status: str  # NO_DRIFT | DRIFT_SUSPECTED | DRIFT_DETECTED
    psi: float   # mean PSI across monitored features
    per_feature_psi: np.ndarray
    n_current: int
    n_reference: int


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute PSI for one feature.

    Uses the reference data's quantiles as bin edges to avoid empty bins
    in the reference distribution. Bins are [0, 1/bins, 2/bins, ...].
    Small epsilon added to avoid log(0).
    """
    eps = 1e-6
    # Quantile bins from reference
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(reference, quantiles)
    # Ensure strictly increasing (handle ties by adding tiny jitter to edges)
    for i in range(1, len(bin_edges)):
        if bin_edges[i] <= bin_edges[i - 1]:
            bin_edges[i] = bin_edges[i - 1] + eps
    # Histogram proportions
    ref_hist, _ = np.histogram(reference, bins=bin_edges)
    cur_hist, _ = np.histogram(current, bins=bin_edges)
    ref_prop = (ref_hist + eps) / (len(reference) + eps * bins)
    cur_prop = (cur_hist + eps) / (len(current) + eps * bins)
    # PSI = sum((cur - ref) * log(cur / ref))
    psi = float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))
    return psi


class DriftDetector:
    def __init__(self, config: DriftConfig, reference_data: np.ndarray):
        """
        Args:
            config: DriftConfig with thresholds and feature subset.
            reference_data: 2D array (n_samples, n_features) of the reference
                population. Only the features in psi_feature_subset (or all if None)
                are stored for PSI computation.
        """
        self.config = config
        self.reference = reference_data
        if config.psi_feature_subset is not None:
            self.ref_features = reference_data[:, config.psi_feature_subset]
        else:
            self.ref_features = reference_data
        self.n_features = self.ref_features.shape[1]
        self._fitted = True

    def compute(self, current_data: np.ndarray) -> DriftResult:
        """Compute drift between reference and current data."""
        if current_data.shape[1] < self.ref_features.shape[1]:
            raise ValueError(
                f"current_data has {current_data.shape[1]} features but "
                f"{self.ref_features.shape[1]} are required")
        if self.config.psi_feature_subset is not None:
            cur_features = current_data[:, self.config.psi_feature_subset]
        else:
            cur_features = current_data

        # Compute per-feature PSI
        per_feature_psi = np.empty(self.n_features, dtype=np.float32)
        for i in range(self.n_features):
            per_feature_psi[i] = compute_psi(
                self.ref_features[:, i],
                cur_features[:, i],
                bins=self.config.psi_bins,
            )
        mean_psi = float(np.mean(per_feature_psi))

        # Determine status
        if mean_psi >= self.config.psi_detected_threshold:
            status = "DRIFT_DETECTED"
        elif mean_psi >= self.config.psi_suspect_threshold:
            status = "DRIFT_SUSPECTED"
        else:
            status = "NO_DRIFT"

        logger.info("drift PSI=%.4f status=%s (n_cur=%d, n_ref=%d)",
                    mean_psi, status, len(current_data), len(self.reference))
        return DriftResult(
            status=status,
            psi=mean_psi,
            per_feature_psi=per_feature_psi,
            n_current=len(current_data),
            n_reference=len(self.reference),
        )