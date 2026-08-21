"""Aggregation: CSV/JSON/comparison tables/plots (no invented values)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.federated.experiments.plots import plot_comparison_bars


def collect_summaries(experiment_dirs: List[Path]) -> List[Dict[str, Any]]:
    """Load metrics/summary.json from each experiment dir (missing values stay None)."""
    rows: List[Dict[str, Any]] = []
    for d in experiment_dirs:
        # Prefer metrics/summary.json (unified) fallback to summary.json
        candidates = [d / "metrics" / "summary.json", d / "summary.json"]
        summary = None
        for c in candidates:
            if c.exists():
                try:
                    summary = json.loads(c.read_text(encoding="utf-8"))
                    break
                except Exception:
                    continue
        if summary is None:
            continue
        # also load config and reproducibility for columns
        cfg = {}
        for cf in [d / "config_resolved.json", d / "config.json"]:
            if cf.exists():
                try:
                    cfg = json.loads(cf.read_text(encoding="utf-8"))
                    break
                except Exception:
                    pass
        exp_id = d.name
        # Try to read metadata for creation time / model etc
        meta = {}
        for mf in [d / "metadata.json", d / "environment.json"]:
            if mf.exists():
                try:
                    meta = json.loads(mf.read_text(encoding="utf-8"))
                    break
                except Exception:
                    pass
        row = _flatten_summary(summary, cfg, exp_id)
        rows.append(row)
    return rows


def _flatten_summary(summary: Dict[str, Any], cfg: Dict[str, Any], exp_id: str) -> Dict[str, Any]:
    """Flatten a summary into a single row dict (missing -> None, not invented)."""
    # experiment block
    exp = summary.get("experiment") or {}
    # FL config
    fl_cfg = exp.get("fl_config") or cfg.get("fl", {}) or {}
    part_cfg = exp.get("partition_config") or cfg.get("partition", {}) or {}
    model_cfg = exp.get("model") or cfg.get("model", {}) or {}
    train_cfg = cfg.get("train", {}) or {}

    final = summary.get("final_global_test_metrics") or summary.get("final_metrics") or summary.get("final") or {}
    if not isinstance(final, dict):
        final = {}
    comm = summary.get("communication") or {}
    totals = comm.get("totals") or {}
    # resource/drift/privacy/security may be top-level
    resource = summary.get("resource")
    drift = summary.get("drift")
    security = summary.get("security") or {}
    if not security and ("attack" in summary or "defense" in summary):
        security = {"attack": summary.get("attack"), "defense": summary.get("defense")}
    privacy = summary.get("privacy")

    row: Dict[str, Any] = {
        "experiment_id": exp_id,
        "name": cfg.get("name") or exp.get("name"),
        "seed": fl_cfg.get("seed") if isinstance(fl_cfg, dict) else None,
        "dataset": cfg.get("data", {}).get("ember_version") if isinstance(cfg.get("data"), dict) else None,
        "algorithm": exp.get("algorithm") or fl_cfg.get("algorithm") or cfg.get("fl", {}).get("algorithm"),
        "clients": fl_cfg.get("num_clients") if isinstance(fl_cfg, dict) else None,
        "client_fraction": fl_cfg.get("client_fraction") if isinstance(fl_cfg, dict) else None,
        "partition_strategy": part_cfg.get("strategy") if isinstance(part_cfg, dict) else None,
        "non_iid_severity": part_cfg.get("severity") if isinstance(part_cfg, dict) else None,
        "model_hidden": model_cfg.get("hidden_layers") if isinstance(model_cfg, dict) else None,
        "learning_rate": fl_cfg.get("learning_rate") if isinstance(fl_cfg, dict) else train_cfg.get("learning_rate"),
        "batch_size": fl_cfg.get("batch_size") if isinstance(fl_cfg, dict) else train_cfg.get("batch_size"),
        "local_epochs": fl_cfg.get("local_epochs") if isinstance(fl_cfg, dict) else train_cfg.get("local_epochs"),
        "fl_rounds": fl_cfg.get("num_rounds") if isinstance(fl_cfg, dict) else None,
        "fedprox_mu": fl_cfg.get("proximal_mu") if isinstance(fl_cfg, dict) else None,
        "personalization_probe_samples": fl_cfg.get("personalized_probe_samples") if isinstance(fl_cfg, dict) else None,
        "final_accuracy": final.get("accuracy"),
        "final_precision": final.get("precision"),
        "final_recall": final.get("recall"),
        "final_f1": final.get("f1"),
        "final_roc_auc": final.get("roc_auc"),
        "training_time_s": summary.get("training_time_s"),
        "total_bytes": totals.get("total_bytes_exchanged") if isinstance(totals, dict) else None,
        "resource_enabled": (resource.get("enabled") if isinstance(resource, dict) else None),
        "drift_enabled": (drift.get("enabled") if isinstance(drift, dict) else None),
        "privacy_enabled": (privacy.get("enabled") if isinstance(privacy, dict) else None),
        "security_attack": (security.get("attack", {}).get("attack_type") if isinstance(security.get("attack"), dict) else None),
        "security_defense": (security.get("defense", {}).get("defense_mode") if isinstance(security.get("defense"), dict) else None),
    }
    # Preserve explicit Nones; do not invent defaults
    return row


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return out_path
    # union of keys, stable order: put core columns first
    core = [
        "experiment_id",
        "name",
        "seed",
        "dataset",
        "algorithm",
        "clients",
        "client_fraction",
        "partition_strategy",
        "non_iid_severity",
        "fl_rounds",
        "final_accuracy",
        "final_f1",
        "final_roc_auc",
        "training_time_s",
        "total_bytes",
    ]
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    ordered = [k for k in core if k in all_keys] + sorted(k for k in all_keys if k not in core)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            # Missing values stay empty (None -> "")
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in ordered})
    return out_path


def write_json(rows: List[Dict[str, Any]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True, default=str)
    return out_path


def write_comparison_table(rows: List[Dict[str, Any]], out_path: Path) -> Path:
    """Markdown comparison table (missing values rendered as empty)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("No experiments found.\n", encoding="utf-8")
        return out_path
    cols = ["experiment_id", "algorithm", "partition_strategy", "final_f1", "final_roc_auc", "training_time_s", "total_bytes"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c)
            if v is None:
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def generate_plots(rows: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    """Generate bar comparison plots for key metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    # Build comparison dict for plots: label -> row
    comp: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        label = r.get("name") or r.get("experiment_id", "exp")
        # disambiguate duplicate labels
        base = label
        i = 1
        while label in comp:
            i += 1
            label = f"{base}-{i}"
        comp[label] = r
    for metric in ("final_f1", "final_roc_auc", "training_time_s"):
        p = plot_comparison_bars(comp, metric, out_dir / f"comparison_{metric}.png", title=f"Comparison: {metric}")
        if p:
            created.append(p)
    return created
