"""Communication efficiency analysis — all measurements are actual serialized bytes.

No arbitrary estimates: upload_bytes and download_bytes come from
``serialize_bytes`` (sum of ``p.nbytes``) in ``src/federated/fl/client.py:50``
and aggregated in ``src/federated/fl/strategy.py:95``. Model parameter count
and serialized size are measured from the actual ``ndarrays_to_parameters``
payload in ``src/federated/fl/server.py:357``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CommunicationRecord:
    """One experiment's actual communication measurements.

    All byte fields are real serialized sizes, not estimates.
    Missing values stay ``None`` (never invented).
    """

    experiment_id: str
    algorithm: Optional[str] = None
    strategy: Optional[str] = None
    n_clients: Optional[int] = None
    n_rounds: Optional[int] = None
    client_fraction: Optional[float] = None
    model_parameter_count: Optional[int] = None
    model_parameter_bytes: Optional[int] = None
    full_model_parameter_count: Optional[int] = None
    full_model_bytes: Optional[int] = None
    upload_bytes: Optional[int] = None
    download_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    bytes_per_round: Optional[float] = None
    bytes_per_client_per_round: Optional[float] = None
    training_time_s: Optional[float] = None
    final_f1: Optional[float] = None
    final_auc: Optional[float] = None
    convergence_round: Optional[int] = None  # first round reaching 95% of final F1
    per_round_bytes: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _convergence_round(rounds: List[Dict[str, Any]], final_f1: Optional[float]) -> Optional[int]:
    """First round where global F1 reaches 95% of final (None if missing)."""
    if final_f1 is None or not rounds:
        return None
    thresh = 0.95 * final_f1
    for r in rounds:
        ge = r.get("global_eval") or {}
        f1 = ge.get("f1")
        if f1 is not None and f1 >= thresh:
            return int(r.get("round"))
    return None


def from_summary(summary: Dict[str, Any], experiment_id: str = "") -> CommunicationRecord:
    """Build a CommunicationRecord from a unified or raw FL summary (no invention)."""
    exp = summary.get("experiment") or {}
    fl_cfg = exp.get("fl_config") or {}
    # Communication block — all actual measured bytes
    comm = summary.get("communication") or {}
    totals = comm.get("totals") or {}
    # Support both unified and legacy keys
    model_cnt = comm.get("model_parameter_count")
    model_bytes = comm.get("model_parameter_bytes")
    # Fallback: derive from per_round if totals missing? No — keep None (honest)
    total_bytes = totals.get("total_bytes_exchanged")
    # If total_bytes still None, try alternative key
    if total_bytes is None:
        total_bytes = summary.get("total_bytes")
    n_rounds = fl_cfg.get("num_rounds")
    if n_rounds is None:
        n_rounds = exp.get("num_rounds") or len(summary.get("rounds") or [])
        if n_rounds == 0:
            n_rounds = None
    n_clients = fl_cfg.get("num_clients")
    if n_clients is None:
        n_clients = exp.get("num_clients")

    # Per-round bytes for tradeoff plots
    per_round = comm.get("per_round") or []

    total_upload = totals.get("total_upload_bytes")
    total_download = totals.get("total_download_bytes")

    # bytes per round = total / rounds (actual, not estimate)
    bpr = None
    if total_bytes is not None and n_rounds:
        try:
            bpr = float(total_bytes) / float(n_rounds)
        except Exception:
            bpr = None
    # bytes per client per round
    bpcpr = None
    if bpr is not None and n_clients:
        try:
            bpcpr = bpr / float(n_clients)
        except Exception:
            pass

    # Final F1 (actual measured)
    final = summary.get("final_global_test_metrics") or summary.get("final_metrics") or summary.get("final") or {}
    if not isinstance(final, dict):
        final = {}
    f1 = final.get("f1")
    auc = final.get("roc_auc")

    conv = _convergence_round(summary.get("rounds") or [], f1)

    return CommunicationRecord(
        experiment_id=experiment_id or summary.get("experiment_id", ""),
        algorithm=exp.get("algorithm") or fl_cfg.get("algorithm"),
        strategy=exp.get("strategy"),
        n_clients=n_clients,
        n_rounds=n_rounds,
        client_fraction=fl_cfg.get("client_fraction"),
        model_parameter_count=model_cnt,
        model_parameter_bytes=model_bytes,
        full_model_parameter_count=comm.get("full_model_parameter_count"),
        full_model_bytes=comm.get("full_model_bytes"),
        upload_bytes=total_upload,
        download_bytes=total_download,
        total_bytes=total_bytes,
        bytes_per_round=bpr,
        bytes_per_client_per_round=bpcpr,
        training_time_s=summary.get("training_time_s"),
        final_f1=f1,
        final_auc=auc,
        convergence_round=conv,
        per_round_bytes=per_round if per_round else None,
    )


def analyze_communication_efficiency(summaries: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Aggregate list of (experiment_id, summary) into efficiency report.

    Returns dict with records, comparisons, tradeoffs — all missing left as None.
    """
    records = [from_summary(s, eid) for eid, s in summaries]
    # Comparison tables
    comparison = compare_algorithms(records)
    tradeoffs = compute_tradeoffs(records)
    return {
        "records": [r.to_dict() for r in records],
        "comparison": comparison,
        "tradeoffs": tradeoffs,
    }


def compare_algorithms(records: List[CommunicationRecord]) -> Dict[str, Any]:
    """Compare FedAvg / FedProx / Personalized under different dimensions."""
    # Group by algorithm
    by_algo: Dict[str, List[CommunicationRecord]] = {}
    for r in records:
        by_algo.setdefault(r.algorithm or "unknown", []).append(r)

    result: Dict[str, Any] = {"by_algorithm": {}, "by_strategy": {}, "by_rounds": {}, "by_clients": {}}

    for algo, lst in by_algo.items():
        # Only include actual values (not invented)
        f1s = [r.final_f1 for r in lst if r.final_f1 is not None]
        tbs = [r.total_bytes for r in lst if r.total_bytes is not None]
        result["by_algorithm"][algo] = {
            "n_experiments": len(lst),
            "mean_f1": float(np.mean(f1s)) if f1s else None,
            "mean_total_bytes": float(np.mean(tbs)) if tbs else None,
            "mean_bytes_per_round": float(np.mean([r.bytes_per_round for r in lst if r.bytes_per_round is not None])) if any(r.bytes_per_round is not None for r in lst) else None,
            "records": [r.experiment_id for r in lst],
        }

    # By Non-IID severity (strategy)
    by_strat: Dict[str, List[CommunicationRecord]] = {}
    for r in records:
        by_strat.setdefault(r.strategy or "unknown", []).append(r)
    for strat, lst in by_strat.items():
        f1s = [r.final_f1 for r in lst if r.final_f1 is not None]
        tbs = [r.total_bytes for r in lst if r.total_bytes is not None]
        result["by_strategy"][strat] = {
            "n_experiments": len(lst),
            "mean_f1": float(np.mean(f1s)) if f1s else None,
            "mean_total_bytes": float(np.mean(tbs)) if tbs else None,
        }

    # By round counts
    by_rounds: Dict[str, List[CommunicationRecord]] = {}
    for r in records:
        key = str(r.n_rounds) if r.n_rounds is not None else "unknown"
        by_rounds.setdefault(key, []).append(r)
    for k, lst in by_rounds.items():
        f1s = [r.final_f1 for r in lst if r.final_f1 is not None]
        tbs = [r.total_bytes for r in lst if r.total_bytes is not None]
        result["by_rounds"][k] = {
            "n_experiments": len(lst),
            "mean_f1": float(np.mean(f1s)) if f1s else None,
            "mean_total_bytes": float(np.mean(tbs)) if tbs else None,
        }

    # By client counts
    by_clients: Dict[str, List[CommunicationRecord]] = {}
    for r in records:
        key = str(r.n_clients) if r.n_clients is not None else "unknown"
        by_clients.setdefault(key, []).append(r)
    for k, lst in by_clients.items():
        f1s = [r.final_f1 for r in lst if r.final_f1 is not None]
        tbs = [r.total_bytes for r in lst if r.total_bytes is not None]
        result["by_clients"][k] = {
            "n_experiments": len(lst),
            "mean_f1": float(np.mean(f1s)) if f1s else None,
            "mean_total_bytes": float(np.mean(tbs)) if tbs else None,
        }

    return result


def compute_tradeoffs(records: List[CommunicationRecord]) -> Dict[str, Any]:
    """Communication vs F1 / convergence / clients / training time (no invention)."""
    # Filter to records where both axes are present
    def pairs(x_attr: str, y_attr: str) -> List[Dict[str, Any]]:
        out = []
        for r in records:
            x = getattr(r, x_attr)
            y = getattr(r, y_attr)
            if x is not None and y is not None:
                out.append({"experiment_id": r.experiment_id, "algorithm": r.algorithm, "strategy": r.strategy, x_attr: x, y_attr: y})
        return out

    return {
        "communication_vs_f1": pairs("total_bytes", "final_f1"),
        "bytes_per_round_vs_f1": pairs("bytes_per_round", "final_f1"),
        "communication_vs_convergence": pairs("total_bytes", "convergence_round"),
        "communication_vs_clients": pairs("total_bytes", "n_clients"),
        "communication_vs_training_time": pairs("total_bytes", "training_time_s"),
        "bytes_per_round_vs_training_time": pairs("bytes_per_round", "training_time_s"),
    }


def to_csv_rows(records: List[CommunicationRecord]) -> List[Dict[str, Any]]:
    """Flatten records to CSV-ready rows (missing stays None for empty field)."""
    return [r.to_dict() for r in records]


def generate_tradeoff_plots(tradeoffs: Dict[str, Any], out_dir: Path) -> List[Path]:
    """Scatter plots for each tradeoff (best-effort, skip if matplotlib missing)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    mapping = {
        "communication_vs_f1": ("total_bytes", "final_f1", "Total Bytes vs F1", "Total Bytes", "F1"),
        "communication_vs_convergence": ("total_bytes", "convergence_round", "Total Bytes vs Convergence Round", "Total Bytes", "Convergence Round"),
        "communication_vs_training_time": ("total_bytes", "training_time_s", "Total Bytes vs Training Time", "Total Bytes", "Training Time (s)"),
        "bytes_per_round_vs_f1": ("bytes_per_round", "final_f1", "Bytes/Round vs F1", "Bytes/Round", "F1"),
    }
    for key, (xcol, ycol, title, xlabel, ylabel) in mapping.items():
        data = tradeoffs.get(key) or []
        if not data:
            continue
        xs = [d[xcol] for d in data if d.get(xcol) is not None and d.get(ycol) is not None]
        ys = [d[ycol] for d in data if d.get(xcol) is not None and d.get(ycol) is not None]
        if not xs or not ys:
            continue
        # Color by algorithm if available
        algos = [d.get("algorithm") or "unknown" for d in data]
        uniq = sorted(set(algos))
        cmap = plt.get_cmap("tab10")
        plt.figure(figsize=(7, 5))
        for idx, algo in enumerate(uniq):
            ax = [x for x, d in zip(xs, data) if (d.get("algorithm") or "unknown") == algo]
            ay = [y for y, d in zip(ys, data) if (d.get("algorithm") or "unknown") == algo]
            if not ax:
                continue
            plt.scatter(ax, ay, label=algo, color=cmap(idx % 10), alpha=0.8)
            for x, y, lbl in zip(ax, ay, [d["experiment_id"] for d in data if (d.get("algorithm") or "unknown") == algo]):
                # lightweight annotation? skip to avoid clutter
                pass
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        if len(uniq) > 1:
            plt.legend()
        plt.tight_layout()
        p = out_dir / f"{key}.png"
        plt.savefig(p, dpi=150)
        plt.close()
        created.append(p)
    return created
