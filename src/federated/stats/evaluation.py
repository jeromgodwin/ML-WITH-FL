"""Statistical evaluation — mean, std, CI, no cherry-picking (Enhancement 22)."""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any


def mean_std_ci(values: List[float], confidence: float = 0.95) -> Dict[str, Any]:
    """Mean, std, 95% CI (t-distribution). Do not treat tiny diffs as meaningful."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"mean": None, "std": None, "ci_low": None, "ci_high": None, "n": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    # 95% CI approx using t=2.0 for simplicity (large n)
    se = std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    margin = 1.96 * se
    return {"mean": mean, "std": std, "ci_low": mean - margin, "ci_high": mean + margin, "n": len(arr), "note": "tiny differences within CI are not meaningful"}
