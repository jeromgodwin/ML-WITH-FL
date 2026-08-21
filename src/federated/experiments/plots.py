"""Plotting utilities for Phase 15 (no invented values)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def plot_f1_per_round(
    rounds: List[Dict[str, Any]],
    out_path: Path,
    title: str = "F1 per round",
) -> Optional[Path]:
    """Line plot of global F1 and avg client F1 per round."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if not rounds:
        return None
    xs = [r.get("round", i + 1) for i, r in enumerate(rounds)]
    # global F1
    g = []
    for r in rounds:
        ge = r.get("global_eval") or {}
        g.append(ge.get("f1"))
    a = [r.get("avg_client_f1") for r in rounds]
    w = [r.get("worst_client_f1") for r in rounds]
    # Filter None for plotting - keep gaps as NaN so line breaks honestly
    def to_float(v):
        return float(v) if v is not None else np.nan

    g = [to_float(v) for v in g]
    a = [to_float(v) for v in a]
    w = [to_float(v) for v in w]

    # If all NaN, skip
    if all(np.isnan(v) for v in g) and all(np.isnan(v) for v in a):
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    if not all(np.isnan(v) for v in g):
        plt.plot(xs, g, marker="o", label="global F1")
    if not all(np.isnan(v) for v in a):
        plt.plot(xs, a, marker="s", label="avg client F1")
    if not all(np.isnan(v) for v in w):
        plt.plot(xs, w, marker="^", label="worst client F1", linestyle="--")
    plt.xlabel("Round")
    plt.ylabel("F1")
    plt.title(title)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_communication(
    rounds: List[Dict[str, Any]],
    out_path: Path,
    title: str = "Bytes per round",
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if not rounds or not any(r.get("bytes_this_round") is not None for r in rounds):
        return None
    xs = [r.get("round", i + 1) for i, r in enumerate(rounds)]
    ys = [r.get("bytes_this_round") or 0 for r in rounds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.bar(xs, ys, color="steelblue")
    plt.xlabel("Round")
    plt.ylabel("Bytes")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_comparison_bars(
    comparison: Dict[str, List[Dict[str, Any]]],
    metric: str,
    out_path: Path,
    title: Optional[str] = None,
) -> Optional[Path]:
    """Bar chart comparison across experiments (missing values omitted)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    # comparison: {label: metrics dict}
    labels = []
    values = []
    for label, row in comparison.items():
        v = row.get(metric) if isinstance(row, dict) else None
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        labels.append(label)
        values.append(fv)
    if not labels:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6, len(labels) * 0.6), 4))
    plt.bar(labels, values, color="teal")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric)
    plt.title(title or f"Comparison: {metric}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
