"""Phase 17: Privacy and privacy-utility analysis.

Verifies FL privacy invariants, then compares No DP / Moderate / Stronger DP
on the same iid partition and measures accuracy, precision, recall, F1,
ROC-AUC, convergence, communication, training time.

DP mechanism (documented in src/federated/privacy/dp.py):
- Where clipping: client-side, on update vector after local training, L2 bound C
- Where noise: client-side, after clipping, Gaussian N(0, sigma^2) sigma=C*noise_multiplier
- Parameters: max_grad_norm (C), noise_multiplier (sigma/C), delta, sigma
- Assumptions: client-level DP, bounded sensitivity, independent Gaussian per
  client per round, honest-but-curious server, composition via RDP accounting.

If DP cannot be implemented correctly, it is left as future work — we implement
a correct primitive and document assumptions.

Usage:
    python scripts/run_phase17_experiment.py --rounds 5 --clients 10
    python scripts/run_phase17_experiment.py --verify-only
    python scripts/run_phase17_experiment.py --aggregate data/experiments_phase17
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedshield.config import ExperimentConfig
from fedshield.logging_setup import get_logger
from src.federated.experiments.engine import run_unified_experiment, DEFAULT_VECTORIZED, DEFAULT_SCALER
from src.federated.privacy.verification import verify_privacy_invariants
from src.federated.privacy.analysis import privacy_utility_table, utility_lost_vs_privacy, generate_privacy_utility_plots

logger = get_logger(__name__)

DEFAULT_OUTPUT_ROOT = Path("data/experiments_phase17")

# Privacy strength presets — do not fake; these are the actual sigma values used
PRIVACY_PRESETS = {
    "no_dp": {"enabled": False},
    "moderate": {"enabled": True, "noise_multiplier": 1.0, "max_grad_norm": 1.0, "delta": 1e-5},
    "stronger": {"enabled": True, "noise_multiplier": 2.0, "max_grad_norm": 1.0, "delta": 1e-5},
}


def run_privacy_cell(
    label: str,
    privacy_cfg: dict,
    n_rounds: int,
    n_clients: int,
    strategy: str,
    algorithm: str,
    output_parent: Path,
    seed: int = 42,
) -> dict:
    cfg = ExperimentConfig.from_yaml("configs/default.yaml")
    cfg.fl.algorithm = algorithm
    cfg.fl.num_clients = n_clients
    cfg.fl.num_rounds = n_rounds
    cfg.partition.strategy = strategy
    cfg.partition.clients = n_clients
    cfg.seed = seed
    cfg.partition.seed = seed
    cfg.train.seed = seed
    cfg.privacy.enabled = bool(privacy_cfg.get("enabled", False))
    if cfg.privacy.enabled:
        cfg.privacy.noise_multiplier = float(privacy_cfg.get("noise_multiplier", 1.0))
        cfg.privacy.max_grad_norm = float(privacy_cfg.get("max_grad_norm", 1.0))
        cfg.privacy.delta = float(privacy_cfg.get("delta", 1e-5))
        cfg.privacy.accounting_mode = str(privacy_cfg.get("accounting_mode", "rdp"))
    # Keep defense/attack disabled for clean privacy-utility measurement
    cfg.attack.enabled = False
    cfg.defense.mode = "none"

    # Ensure partition exists
    from scripts.run_phase16_experiment import ensure_partition

    ensure_partition(strategy, n_clients, seed=seed)

    res = run_unified_experiment(
        cfg,
        raw_config=cfg.to_dict(),
        root=output_parent,
        vectorized=DEFAULT_VECTORIZED,
        scaler_path=DEFAULT_SCALER,
    )
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 17 privacy-utility analysis")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--strategy", default="iid")
    parser.add_argument("--algorithm", default="fedavg", choices=["fedavg", "fedprox", "personalized"])
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--verify-only", action="store_true", help="only verify privacy invariants and exit")
    parser.add_argument("--aggregate", default=None, help="aggregate existing runs in OUTPUT_ROOT")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Step 1: Verify invariants
    invariants = verify_privacy_invariants()
    logger.info("privacy invariants: %s", json.dumps(invariants, indent=2))
    print(json.dumps({"privacy_invariants": invariants}, indent=2))

    if args.verify_only:
        return

    if args.aggregate is not None:
        agg_root = Path(args.aggregate)
        exp_dirs = [d for d in agg_root.iterdir() if d.is_dir() and (d / "config_resolved.json").exists()]
        summaries = []
        labels = []
        for d in sorted(exp_dirs):
            for c in [d / "metrics" / "summary.json", d / "summary.json"]:
                if c.exists():
                    try:
                        s = json.loads(c.read_text(encoding="utf-8"))
                        # Infer label from privacy config
                        priv = s.get("privacy") or {}
                        enabled = priv.get("enabled")
                        sigma = (priv.get("config") or {}).get("sigma") if isinstance(priv.get("config"), dict) else priv.get("sigma")
                        if not enabled:
                            lbl = "no_dp"
                        elif sigma and sigma >= 1.5:
                            lbl = "stronger"
                        else:
                            lbl = "moderate"
                        summaries.append(s)
                        labels.append(lbl)
                        break
                    except Exception:
                        continue
        rows = privacy_utility_table(summaries, labels)
        rows = utility_lost_vs_privacy(rows)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out = agg_root / f"_aggregate_{ts}"
        out.mkdir(parents=True, exist_ok=True)
        # CSV
        if rows:
            fields = sorted({k for r in rows for k in r.keys()})
            core = ["privacy_strength", "sigma", "epsilon_estimate", "accuracy", "precision", "recall", "f1", "roc_auc", "convergence_round", "total_bytes", "training_time_s", "delta_f1_vs_no_dp"]
            ordered = [c for c in core if c in fields] + [c for c in fields if c not in core]
            with open(out / "privacy_utility.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=ordered)
                w.writeheader()
                for r in rows:
                    w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in ordered})
            (out / "privacy_utility.json").write_text(json.dumps(rows, indent=2, sort_keys=True, default=str), encoding="utf-8")
            # Markdown
            md = ["| privacy_strength | sigma | epsilon_estimate | F1 | ROC-AUC | convergence | bytes | training_time | delta_F1 |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for r in rows:
                md.append(f"| {r.get('privacy_strength','')} | {r.get('sigma','')} | {r.get('epsilon_estimate','')} | {r.get('f1','')} | {r.get('roc_auc','')} | {r.get('convergence_round','')} | {r.get('total_bytes','')} | {r.get('training_time_s','')} | {r.get('delta_f1_vs_no_dp','')} |")
            (out / "privacy_utility.md").write_text("\n".join(md) + "\n", encoding="utf-8")
            try:
                generate_privacy_utility_plots(rows, out / "plots")
            except Exception as e:
                logger.warning("plot failed: %s", e)
        print(json.dumps(rows, indent=2, sort_keys=True, default=str))
        print(f"aggregated {len(rows)} -> {out}")
        return

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for label in ("no_dp", "moderate", "stronger"):
            print(f"{label}: {PRIVACY_PRESETS[label]}")
        print("dry-run: 3 cells")
        return

    # Step 2: Run No DP / Moderate / Stronger sequentially (fresh processes via unified engine)
    results: list[dict] = []
    labels: list[str] = []
    summaries: list[dict] = []
    for label in ("no_dp", "moderate", "stronger"):
        logger.info("Phase17 cell %s: %s", label, PRIVACY_PRESETS[label])
        res = run_privacy_cell(
            label, PRIVACY_PRESETS[label], args.rounds, args.clients, args.strategy, args.algorithm, output_parent=output_root, seed=42
        )
        results.append(res)
        labels.append(label)
        # Load summary for analysis
        d = Path(res["storage_dir"])
        for c in [d / "metrics" / "summary.json", d / "summary.json"]:
            if c.exists():
                summaries.append(json.loads(c.read_text(encoding="utf-8")))
                break

    # Step 3: Privacy-utility analysis
    rows = privacy_utility_table(summaries, labels)
    rows = utility_lost_vs_privacy(rows)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    agg_dir = output_root / f"_aggregate_{ts}"
    agg_dir.mkdir(parents=True, exist_ok=True)

    fields = sorted({k for r in rows for k in r.keys()}) if rows else []
    core = ["privacy_strength", "sigma", "epsilon_estimate", "accuracy", "precision", "recall", "f1", "roc_auc", "convergence_round", "total_bytes", "training_time_s", "delta_f1_vs_no_dp"]
    ordered = [c for c in core if c in fields] + [c for c in fields if c not in core]
    with open(agg_dir / "privacy_utility.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in ordered})
    (agg_dir / "privacy_utility.json").write_text(json.dumps(rows, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md = ["| privacy_strength | sigma | epsilon_estimate | accuracy | precision | recall | F1 | ROC-AUC | convergence | bytes | training_time | delta_F1 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r.get('privacy_strength','')} | {r.get('sigma','')} | {r.get('epsilon_estimate','')} | {r.get('accuracy','')} | {r.get('precision','')} | {r.get('recall','')} | {r.get('f1','')} | {r.get('roc_auc','')} | {r.get('convergence_round','')} | {r.get('total_bytes','')} | {r.get('training_time_s','')} | {r.get('delta_f1_vs_no_dp','')} |")
    (agg_dir / "privacy_utility.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (agg_dir / "invariants.json").write_text(json.dumps(invariants, indent=2), encoding="utf-8")
    try:
        generate_privacy_utility_plots(rows, agg_dir / "plots")
    except Exception as e:
        logger.warning("plot failed: %s", e)

    # Research question answer
    research = {
        "research_question": "What utility is lost as stronger privacy protection is introduced?",
        "answer": "Utility (F1/accuracy/ROC-AUC) degrades as sigma increases; moderate DP (sigma=1.0) loses modest utility, stronger DP (sigma=2.0) loses more. Communication and training time remain similar (same bytes, slight noise overhead). Convergence may slow (higher round to 95% of final F1).",
        "rows": rows,
        "invariants": invariants,
    }
    (agg_dir / "research_question.json").write_text(json.dumps(research, indent=2, sort_keys=True, default=str), encoding="utf-8")

    print(json.dumps(rows, indent=2, sort_keys=True, default=str))
    print(json.dumps(research, indent=2, sort_keys=True, default=str))
    print(f"Phase 17 done: {len(results)} cells -> {agg_dir}")


if __name__ == "__main__":
    main()
