"""Experimental robust aggregation strategies (Phase 14).

Implemented exactly:

- coordinate_wise_median: the aggregated parameter vector is the per-
  coordinate median across clients (each weight's median value).
- trimmed_mean: per-coordinate mean computed over clients after trimming
  the ``trim_frac`` largest and smallest values per coordinate.

Both operate on the raw client parameter matrices (concatenated updates).
They are intentionally simple and do NOT claim to make FedShield immune
to poisoning; they are experimental alternatives to weighted averaging.
"""

from __future__ import annotations

from typing import List

import numpy as np


def coordinate_wise_median(updates: List[np.ndarray]) -> np.ndarray:
    """Per-coordinate median across client update vectors."""
    if not updates:
        raise ValueError("no updates to aggregate")
    matrix = np.stack(updates, axis=0)
    return np.median(matrix, axis=0)


def trimmed_mean(updates: List[np.ndarray], trim_frac: float = 0.2) -> np.ndarray:
    """Per-coordinate mean over clients after trimming extremes.

    For each coordinate, the ``trim_frac`` largest and smallest values are
    removed before averaging (values per side rounded down to whole counts).
    """
    if not updates:
        raise ValueError("no updates to aggregate")
    if not 0.0 <= trim_frac < 0.5:
        raise ValueError(f"trim_frac must be in [0, 0.5), got {trim_frac}")
    matrix = np.stack(updates, axis=0)
    n = matrix.shape[0]
    k = int(trim_frac * n)
    if k == 0:
        return matrix.mean(axis=0)
    sorted_m = np.sort(matrix, axis=0)
    return sorted_m[k : n - k].mean(axis=0)
