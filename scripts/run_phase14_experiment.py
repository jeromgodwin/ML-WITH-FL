"""Phase 14 experiment: poisoning attacks and malicious-client defense.

CONTROLLED CYBERSECURITY EXPERIMENT. Simulates abnormal model updates ONLY —
no real malware, no operational attack tooling. The purpose is to measure
attack impact, detection capability, and mitigation capability.

Matrix (all on the SAME iid partition + official test set):

  defense \\ attack       none            scaled_update       label_flip
  baseline (FedAvg)       (clean ref)     impact ref           impact ref
  clipping                   -               defended             defended
  anomaly detection          -               defended             defended
  model validation           -               defended             defended
  robust median              -               defended             defended
  robust trimmed             -               defended             defended

For every run we record: global test F1, worst-client F1, validation F1,
communication bytes, round time, per-client anomaly classifications (for
detection rate / false-positive rate), clipping records, and validation
gate decisions.

Honest limitation: "FedShield evaluates selected poisoning-defense
mechanisms under controlled simulated attacks." — it does NOT claim to
prevent poisoning attacks.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from fedshield.config import ExperimentConfig  # noqa: E402
from fedshield.logging_setup import get_logger  # noqa: E402
from src.federated.data.dataset import load_split  # noqa: E402
from src.federated.fl.server import run_fl_experiment  # noqa: E402

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("configs/default.yaml")
DEFAULT_VECTORIZED = Path("data/ember_2018_2/vectorized")
DEFAULT_SCALER = Path("data/ember_2018_2/artifacts/scaler.joblib")
PARTITION_ROOT = Path("data")
OUTPUT_ROOT = Path("data/fl/phase14")

ATTACKS = ["none", "scaled_update", "label_flip"]
DEFENSES = ["baseline", "clipping", "anomaly", "validation",
            "robust_median", "robust_trimmed"]


def detection_fp_rates(res: Dict[str, Any]) -> Dict[str, Any]:
    """Detection rate + false-positive rate from per-round anomaly records.

    Detection rate: fraction of malicious update-opportunities classified
    SUSPICIOUS or HIGHLY_ANOMALOUS. FP rate: fraction of honest
    update-opportunities flagged. For the baseline (no defense) both are 0.
    """
    attack = res.get("attack", {})
    malicious = set(int(c) for c in attack.get("malicious_cids", []))
    n_mal_opps = 0
    n_mal_flagged = 0
    n_honest_opps = 0
    n_honest_flagged = 0
    n_excluded = 0
    for r in res.get("rounds", []):
        defense = r.get("defense", {})
        anomaly = defense.get("anomaly") or []
        for a in anomaly:
            cid = int(a["cid"])
            flagged = a["classification"] != "NORMAL"
            if cid in malicious:
                n_mal_opps += 1
                n_mal_flagged += int(flagged)
            else:
                n_honest_opps += 1
                n_honest_flagged += int(flagged)
        n_excluded += defense.get("n_excluded_anomalous", 0)
    return {
        "detection_rate": round(n_mal_flagged / n_mal_opps, 4)
        if n_mal_opps else 0.0,
        "false_positive_rate": round(n_honest_flagged / n_honest_opps, 4)
        if n_honest_opps else 0.0,
        "n_malicious_updates": n_mal_opps,
        "n_malicious_flagged": n_mal_flagged,
        "n_honest_updates": n_honest_opps,
        "n_honest_flagged": n_honest_flagged,
        "n_updates_excluded": n_excluded,
    }


def run_cell(
    cfg: ExperimentConfig,
    partition_dir: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scale_inv: np.ndarray,
    defense: str,
    attack: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run one (defense, attack) cell and return the measured result.

    Resumes: if the cell's summary.json already exists, the saved result is
    loaded instead of re-running (used when the batch is interrupted).
    """
    if (out_dir / "summary.json").exists():
        saved = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        final = saved.get("final_global_test_metrics") or {}
        bytes_total = saved.get("communication", {}).get("totals", {}).get(
            "total_bytes_exchanged", 0)
        if final.get("f1") is not None and bytes_total > 0:
            logger.info("cell %s/%s already done; resuming from saved result",
                        defense, attack)
            return saved
        logger.warning("cell %s/%s summary is incomplete (f1=%s bytes=%s); "
                       "re-running", defense, attack, final.get("f1"), bytes_total)
    run_cfg = cfg.with_overrides(
        **{
            "attack.enabled": attack != "none",
            "attack.attack_type": attack,
            "defense.mode": {
                "baseline": "none",
                "clipping": "clipping",
                "anomaly": "anomaly",
                "validation": "validation",
                "robust_median": "robust_median",
                "robust_trimmed": "robust_trimmed",
            }[defense],
        }
    )
    if defense == "clipping":
        run_cfg.defense.clip_norm = 0.05
    logger.info("=== cell: defense=%s attack=%s ===", defense, attack)
    res = run_fl_experiment(
        run_cfg, partition_dir, X_train, y_train, X_test, y_test,
        scale_inv=scale_inv, output_dir=out_dir, seed=run_cfg.seed)
    return res


def collect_metrics(res: Dict[str, Any]) -> Dict[str, Any]:
    """Key metrics + detection/FP rates for one cell."""
    final = res.get("final_global_test_metrics") or {}
    rounds = res.get("rounds", [])
    worst = [r.get("worst_client_f1") for r in rounds
             if r.get("worst_client_f1") is not None]
    final_worst = worst[-1] if worst else None
    defense = res.get("defense", {})
    validation = defense.get("validation") or {}
    return {
        "global_test_f1": final.get("f1"),
        "global_test_auc": final.get("roc_auc"),
        "worst_client_f1": final_worst,
        "validation_trusted_f1": validation.get("trusted_f1"),
        "validation_n_reject": validation.get("n_reject", 0),
        "total_bytes": res["communication"]["totals"]["total_bytes_exchanged"],
        "training_time_s": res["training_time_s"],
        "detection": detection_fp_rates(res),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--defense", choices=DEFENSES, default=None)
    parser.add_argument("--attack", choices=ATTACKS, default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(DEFAULT_CONFIG)
    cfg.fl.num_rounds = 5
    cfg.fl.algorithm = "fedavg"
    cfg.fl.num_clients = 10

    scaler = joblib.load(DEFAULT_SCALER)
    scale_inv = np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float32)
    scale_inv = (1.0 / scale_inv).astype(np.float32)

    X_train, y_train = load_split(DEFAULT_VECTORIZED, "train")
    X_test, y_test = load_split(DEFAULT_VECTORIZED, "test")

    base_partition = PARTITION_ROOT / f"iid-c{cfg.fl.num_clients}-s{cfg.seed}"
    partition_dir = base_partition
    if not (base_partition / "client_indices.npz").exists():
        raise FileNotFoundError(f"partition not found: {base_partition}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Rebuild the matrix from the on-disk cell summaries so a partial batch
    # (one cell per process, interrupted runs) is fully represented.
    matrix: Dict[str, Dict[str, Any]] = {}
    clean = None
    for defense in DEFENSES:
        matrix[defense] = {}
        for attack in ATTACKS:
            if args.defense is not None and args.defense != defense:
                continue
            if args.attack is not None and args.attack != attack:
                continue
            cell_dir = OUTPUT_ROOT / f"{defense}__{attack}"
            res = run_cell(cfg, partition_dir, X_train, y_train,
                           X_test, y_test, scale_inv, defense, attack,
                           cell_dir)
            metrics = collect_metrics(res)
            metrics["cell"] = {"defense": defense, "attack": attack}
            matrix[defense][attack] = metrics
            if defense == "baseline" and attack == "none":
                clean = metrics
            logger.info("%s/%s: F1=%.4f worst=%.4f bytes=%d det=%.4f fp=%.4f",
                        defense, attack,
                        metrics["global_test_f1"] or -1,
                        metrics["worst_client_f1"] or -1,
                        metrics["total_bytes"],
                        metrics["detection"]["detection_rate"],
                        metrics["detection"]["false_positive_rate"])

    if args.defense is not None or args.attack is not None:
        # Single-cell invocation: merge with the existing on-disk summaries
        # and rewrite the comparison so a sequence of single-cell processes
        # still produces the full matrix at the end.
        for defense in DEFENSES:
            for attack in ATTACKS:
                cell_dir = OUTPUT_ROOT / f"{defense}__{attack}"
                summary_path = cell_dir / "summary.json"
                if not summary_path.exists():
                    continue
                res = json.loads(summary_path.read_text(encoding="utf-8"))
                metrics = collect_metrics(res)
                metrics["cell"] = {"defense": defense, "attack": attack}
                matrix.setdefault(defense, {})[attack] = metrics
                if defense == "baseline" and attack == "none":
                    clean = metrics

    # Attack impact: F1 drop vs the clean baseline (baseline+none).
    summary = {
        "experiment": "phase14_poisoning_defense",
        "note": "controlled simulated attacks only; FedShield evaluates "
                "selected poisoning-defense mechanisms under controlled "
                "simulated attacks — it does not claim to prevent them.",
        "attack_types": ATTACKS,
        "defense_modes": DEFENSES,
        "n_malicious_clients": 2,
        "matrix": matrix,
        "attack_impact": {
            defense: {
                attack: {
                    "f1_delta": round(
                        (metrics.get("global_test_f1") or 0)
                        - (clean.get("global_test_f1") or 0), 4)
                    if attack != "none" else 0.0,
                }
                for attack, metrics in rows.items()
            }
            for defense, rows in matrix.items()
        },
        "security_limitation": "FedShield evaluates selected poisoning-defense "
                              "mechanisms under controlled simulated attacks.",
    }
    (OUTPUT_ROOT / "comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()