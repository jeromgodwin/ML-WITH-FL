"""Federated experiment runner over saved Phase-4 partitions (Phase 9, Phase 10).

Consumes the partitions built by scripts/build_clients.py
(data/<strategy>-c<clients>-s<seed>/) — no partitioning happens here.
Runs FedAvg or FedProx on the official vectorized matrix + scaler and writes
per-strategy results (rounds.jsonl, summary.json, config.json) under data/fl/.

FedProx (--algorithm fedprox, --mu N) reuses the identical partition, model,
optimizer, seeds and hyperparameters as FedAvg — only the algorithm differs,
so the two are directly comparable.

Personalized FL (--algorithm personalized, Phase 11) is FedPer-style: clients
keep a personal head and only the shared body is aggregated; the global test
evaluation uses a server-side probe head (--probe-samples/--probe-epochs).

Usage:
    python scripts/run_fl.py                          # iid..severe x fedavg
    python scripts/run_fl.py --algorithm fedprox --mu 0.1
    python scripts/run_fl.py --algorithm personalized
    python scripts/run_fl.py --strategies iid,severe --rounds 10 --local-epochs 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from fedshield.config import ExperimentConfig  # noqa: E402
from fedshield.logging_setup import get_logger  # noqa: E402
from src.federated.fl.server import run_fl_experiment  # noqa: E402
from src.federated.data.dataset import load_split  # noqa: E402

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("configs/default.yaml")
DEFAULT_VECTORIZED = Path("data/ember_2018_2/vectorized")
DEFAULT_SCALER = Path("data/ember_2018_2/artifacts/scaler.joblib")
DEFAULT_PARTITION_ROOT = Path("data")
DEFAULT_OUTPUT_ROOT = Path("data/fl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FedAvg/FedProx/Personalized over Phase-4 partitions")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--strategies", default="iid,mild,moderate,severe",
                        help="comma-separated partition strategies")
    parser.add_argument("--algorithm", default="fedavg",
                        choices=["fedavg", "fedprox", "personalized"])
    parser.add_argument("--mu", type=float, default=None,
                        help="FedProx proximal coefficient (fedprox only)")
    parser.add_argument("--probe-samples", type=int, default=None,
                        help="probe-head training sample size (personalized only)")
    parser.add_argument("--probe-epochs", type=int, default=None,
                        help="probe-head training epochs (personalized only)")
    parser.add_argument("--head-epochs", type=int, default=None,
                        help="FedRep phase-A head adaptation epochs (personalized only)")
    parser.add_argument("--head-lr", type=float, default=None,
                        help="FedRep phase-A head learning rate (personalized only)")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--partition-root", default=str(DEFAULT_PARTITION_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--vectorized", default=str(DEFAULT_VECTORIZED))
    parser.add_argument("--scaler", default=str(DEFAULT_SCALER))
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.rounds:
        cfg.fl.num_rounds = args.rounds
    if args.local_epochs:
        cfg.train.local_epochs = args.local_epochs
    if args.lr:
        cfg.train.learning_rate = args.lr
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.fraction:
        cfg.fl.client_fraction = args.fraction
    if args.clients:
        cfg.fl.num_clients = args.clients
    if args.seed:
        cfg.seed = args.seed
    cfg.fl.algorithm = args.algorithm
    if args.mu is not None:
        cfg.fl.proximal_mu = args.mu
    if args.probe_samples is not None:
        cfg.fl.personalized_probe_samples = args.probe_samples
    if args.probe_epochs is not None:
        cfg.fl.personalized_probe_epochs = args.probe_epochs
    if args.head_epochs is not None:
        cfg.fl.personalized_head_epochs = args.head_epochs
    if args.head_lr is not None:
        cfg.fl.personalized_head_learning_rate = args.head_lr

    vectorized = Path(args.vectorized)
    scaler = joblib.load(Path(args.scaler))
    scale_inv = np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float32)
    scale_inv = (1.0 / scale_inv).astype(np.float32)

    X_train, y_train = load_split(vectorized, "train")
    X_test, y_test = load_split(vectorized, "test")

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    partition_root = Path(args.partition_root)
    output_root = Path(args.output_root)
    results_summary = []
    t_all = time.perf_counter()
    for strategy in strategies:
        partition_dir = partition_root / f"{strategy}-c{cfg.fl.num_clients}-s{cfg.seed}"
        if not partition_dir.exists():
            logger.error("partition not found: %s (run scripts/build_clients.py first)",
                         partition_dir)
            sys.exit(1)
        out = output_root / f"{strategy}-{args.algorithm}"
        logger.info("=== %s experiment: %s (partition %s, mu=%s) ===",
                    args.algorithm, strategy, partition_dir, cfg.fl.proximal_mu)
        res = run_fl_experiment(
            cfg, partition_dir, X_train, y_train, X_test, y_test,
            scale_inv=scale_inv, output_dir=out, seed=cfg.seed)
        final = res["final_global_test_metrics"] or {}
        results_summary.append({
            "algorithm": args.algorithm,
            "proximal_mu": cfg.fl.proximal_mu,
            "strategy": strategy,
            "rounds": res["experiment"]["fl_config"]["num_rounds"],
            "final": final,
            "avg_client_f1_last": res["rounds"][-1].get("avg_client_f1"),
            "worst_client_f1_last": res["rounds"][-1].get("worst_client_f1"),
            "client_f1_variance_last": res["rounds"][-1].get("client_f1_variance"),
            "total_bytes_exchanged": res["communication"]["totals"]["total_bytes_exchanged"],
            "training_time_s": res["training_time_s"],
        })
        logger.info("=== %s/%s done: %s ===", args.algorithm, strategy, final)

    (output_root / "results_summary.json").write_text(
        json.dumps({"strategies": results_summary,
                    "total_time_s": round(time.perf_counter() - t_all, 3)},
                   indent=2), encoding="utf-8")
    print(json.dumps(results_summary, indent=2))


if __name__ == "__main__":
    main()