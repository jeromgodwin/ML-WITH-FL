"""Privacy-utility analysis (Phase 17).

Compares No DP / Moderate / Stronger DP on the same partition/algorithm and
measures accuracy, precision, recall, F1, ROC-AUC, convergence, communication,
training time. Produces a privacy-vs-utility table and plots without inventing
missing values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _get_metric(summary: Dict[str, Any], key: str) -> Optional[float]:
    final = summary.get("final_global_test_metrics") or summary.get("final_metrics") or summary.get("final") or {}
    if not isinstance(final, dict):
        return None
    v = final.get(key)
    return float(v) if v is not None else None


def _convergence_round(summary: Dict[str, Any]) -> Optional[int]:
    rounds = summary.get("rounds") or []
    final = summary.get("final_global_test_metrics") or summary.get("final_metrics") or {}
    f1 = final.get("f1") if isinstance(final, dict) else None
    if f1 is None or not rounds:
        return None
    thresh = 0.95 * float(f1)
    for r in rounds:
        ge = r.get("global_eval") or {}
        rf1 = ge.get("f1")
        if rf1 is not None and rf1 >= thresh:
            return int(r.get("round"))
    return None


def privacy_utility_table(
    summaries: List[Dict[str, Any]],
    labels: List[str],
) -> List[Dict[str, Any]]:
    """Build rows for No DP / Moderate / Stronger comparison.

    summaries[i] corresponds to labels[i] (e.g., 'no_dp', 'moderate', 'stronger').
    Missing values stay None.
    """
    rows: List[Dict[str, Any]] = []
    for label, summary in zip(labels, summaries):
        privacy = summary.get("privacy") or {}
        comm = summary.get("communication") or {}
        totals = comm.get("totals") or {}
        row = {
            "privacy_strength": label,
            "privacy_enabled": privacy.get("enabled"),
            "noise_multiplier": (privacy.get("config") or {}).get("noise_multiplier") if isinstance(privacy.get("config"), dict) else privacy.get("noise_multiplier"),
            "max_grad_norm": (privacy.get("config") or {}).get("max_grad_norm") if isinstance(privacy.get("config"), dict) else privacy.get("max_grad_norm"),
            "sigma": (privacy.get("config") or {}).get("sigma") if isinstance(privacy.get("config"), dict) else privacy.get("sigma"),
            "epsilon_estimate": privacy.get("epsilon_estimate"),
            "delta": privacy.get("delta") if "delta" in privacy else (privacy.get("config") or {}).get("delta") if isinstance(privacy.get("config"), dict) else None,
            "accuracy": _get_metric(summary, "accuracy"),
            "precision": _get_metric(summary, "precision"),
            "recall": _get_metric(summary, "recall"),
            "f1": _get_metric(summary, "f1"),
            "roc_auc": _get_metric(summary, "roc_auc"),
            "convergence_round": _convergence_round(summary),
            "total_bytes": totals.get("total_bytes_exchanged"),
            "training_time_s": summary.get("training_time_s"),
        }
        # Also include per-round count if available
        exp = summary.get("experiment") or {}
        fl_cfg = exp.get("fl_config") or {}
        row["n_rounds"] = fl_cfg.get("num_rounds")
        row["n_clients"] = fl_cfg.get("num_clients")
        rows.append(row)
    return rows


def utility_lost_vs_privacy(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute utility loss relative to No DP baseline (first row with 'no_dp' or lowest sigma).

    Returns rows with delta columns; missing baseline stays None (no invention).
    """
    if not rows:
        return []
    # Find baseline: label == 'no_dp' or first
    baseline = None
    for r in rows:
        if str(r.get("privacy_strength")).lower() in ("no_dp", "none", "no dp"):
            baseline = r
            break
    if baseline is None:
        baseline = rows[0]
    base_f1 = baseline.get("f1")
    base_acc = baseline.get("accuracy")
    base_auc = baseline.get("roc_auc")
    out: List[Dict[str, Any]] = []
    for r in rows:
        delta_f1 = (r["f1"] - base_f1) if r.get("f1") is not None and base_f1 is not None else None
        delta_acc = (r["accuracy"] - base_acc) if r.get("accuracy") is not None and base_acc is not None else None
        delta_auc = (r["roc_auc"] - base_auc) if r.get("roc_auc") is not None and base_auc is not None else None
        out.append({
            **r,
            "delta_f1_vs_no_dp": delta_f1,
            "delta_accuracy_vs_no_dp": delta_acc,
            "delta_roc_auc_vs_no_dp": delta_auc,
        })
    return out


def generate_privacy_utility_plots(rows: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    """Plot F1 vs sigma and utility loss vs sigma (best-effort)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    # Filter rows with sigma and f1 (xhigh: deduplicate no_dp already at sigma 0)
    pts = [(r.get("sigma"), r.get("f1"), str(r.get("privacy_strength"))) for r in rows if r.get("sigma") is not None and r.get("f1") is not None]
    # Include No DP as sigma=0 if not already present
    has_no_dp = any(p[0] == 0.0 for p in pts)
    if not has_no_dp:
        for r in rows:
            if str(r.get("privacy_strength")).lower() in ("no_dp", "none") and r.get("f1") is not None:
                pts.append((0.0, r["f1"], "no_dp"))
                break
    # Deduplicate no_dp duplicate if already present
    # Use first occurrence
    if not pts:
        return []
    # Plot F1 vs sigma
    pts_sorted = sorted(pts, key=lambda x: x[0])
    xs = [p[0] for p in pts_sorted]
    ys = [p[1] for p in pts_sorted]
    labels = [p[2] for p in pts_sorted]
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys, marker="o", linestyle="-", color="teal")
    for x, y, lbl in zip(xs, ys, labels):
        plt.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    plt.xlabel("Sigma (noise scale = noise_multiplier * C)  [0 = No DP]")
    plt.ylabel("Final F1")
    plt.title("Privacy-Utility Tradeoff: F1 vs Sigma")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "privacy_vs_f1.png"
    plt.savefig(p, dpi=150)
    plt.close()
    created.append(p)

    # Plot delta F1 vs sigma
    # Compute baseline delta
    base = None
    for r in rows:
        if str(r.get("privacy_strength")).lower() in ("no_dp", "none"):
            base = r.get("f1")
            break
    if base is not None:
        pts2 = [(r.get("sigma", 0.0) if str(r.get("privacy_strength")).lower() not in ("no_dp", "none") else 0.0, (r.get("f1") - base) if r.get("f1") is not None else None, r.get("privacy_strength")) for r in rows if r.get("f1") is not None]
        xs2 = [p[0] for p in pts2 if p[1] is not None]
        ys2 = [p[1] for p in pts2 if p[1] is not None]
        if xs2 and ys2:
            plt.figure(figsize=(7, 4.5))
            plt.plot(xs2, ys2, marker="s", linestyle="--", color="crimson")
            plt.axhline(0, color="gray", linestyle=":", linewidth=1)
            plt.xlabel("Sigma")
            plt.ylabel("Delta F1 vs No DP")
            plt.title("Utility Loss vs Privacy Strength")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            p2 = out_dir / "utility_loss_vs_sigma.png"
            plt.savefig(p2, dpi=150)
            plt.close()
            created.append(p2)
    return created
