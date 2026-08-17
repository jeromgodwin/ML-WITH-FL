"""Build federated client data partitions (non-IID simulation) on EMBER 2018_2.

Clients are index-only: absolute row indices into the labeled train pool; no
sample vectors are copied. The official 200k-row test split never participates.

Usage:
    python scripts/build_clients.py --strategy all
    python scripts/build_clients.py --strategy severe --clients 10 --seed 42

Outputs under data/ember_2018_2/clients/<strategy>-c<clients>-s<seed>/
  client_indices.npz   c<i>_train / c<i>_val / pool (absolute indices)
  families.npz         malware family label per pool row (when strategy=family_skew)
  manifest.json        config + isolation verification + divergence
  report.json          per-client class/family distributions + divergence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from fedshield.logging_setup import get_logger  # noqa: E402
from src.federated.data.dataset import iter_metadata  # noqa: E402
from src.federated.data.partition import (  # noqa: E402
    STRATEGIES, ClientPartitionConfig, build_client_partition, save_partition,
)
from src.federated.data.split import load_split_indices  # noqa: E402

logger = get_logger(__name__)

FIXED_SEVERITY = {
    "mild": "mild", "moderate": "moderate", "severe": "severe",
    "label_skew": 0.1, "quantity_skew": 0.3, "combined_severe": 0.1,
    "iid": "moderate", "family_skew": "moderate",
}


def load_family_labels(vectorized_dir: Path, n_train_rows: int) -> np.ndarray:
    """avclass family per train row ('' when absent); aligned with X_train rows."""
    labels = np.full(n_train_rows, "", dtype=object)
    row = 0
    for record in iter_metadata(vectorized_dir, "train"):
        labels[row] = record.get("avclass") or ""
        row += 1
    if row != n_train_rows:
        raise ValueError(f"meta has {row} rows, expected {n_train_rows}")
    return labels


def build_one(root: Path, strategy: str, cfg: ClientPartitionConfig,
              vectorized_dir: Path, indices_path: Path, n_train_rows: int,
              with_families: bool):
    y_train = np.load(vectorized_dir / "y_train.npy", mmap_mode="r")
    indices = load_split_indices(indices_path)
    labeled_train = indices["train_idx"][
        (y_train[indices["train_idx"]] == 0) | (y_train[indices["train_idx"]] == 1)]

    family_labels = None
    if with_families:
        family_labels = load_family_labels(vectorized_dir, n_train_rows)

    partition = build_client_partition(y_train, cfg, family_labels=family_labels,
                                       pool_idx=labeled_train)
    out_dir = root / f"{strategy}-c{cfg.clients}-s{cfg.seed}"
    report = save_partition(out_dir, partition, y_train,
                            family_labels if with_families else None)
    d = report["divergence"]
    logger.info("  %s: pool=%d cv=%.3f max/min=%.1f kl_w=%.3f fam_ent=%s",
                strategy, report["pool"]["n"], d["count_cv"],
                d["count_max_min_ratio"], d["class_kl_mean_weighted"], d["family_entropy_mean"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EMBER federated client partitions")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--strategy", default="all", help="one of " + ",".join(STRATEGIES) + " or 'all'")
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--severity", default=None, help="mild|moderate|severe or alpha float")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--min-samples", type=int, default=5000)
    parser.add_argument("--test-strategy", default="global_test", choices=("global_test", "per_client"))
    parser.add_argument("--max-samples", type=int, default=None, help="cap on the labeled pool")
    parser.add_argument("--families", action="store_true",
                        help="attach family labels (requires strategy family_skew for the report)")
    args = parser.parse_args()

    root = Path(args.data_dir)
    vectorized_dir = root / "ember_2018_2" / "vectorized"
    indices_path = root / "ember_2018_2" / "artifacts" / "split_indices.npz"
    if not (vectorized_dir / "X_train.npy").exists():
        logger.error("vectorized train data missing: run scripts/prepare_ember.py first")
        sys.exit(1)

    n_train_rows = int(np.load(vectorized_dir / "X_train.npy", mmap_mode="r").shape[0])
    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    if args.strategy != "all" and args.strategy not in STRATEGIES:
        parser.error(f"unknown strategy {args.strategy!r}")

    for strategy in strategies:
        severity = args.severity or FIXED_SEVERITY[strategy]
        cfg = ClientPartitionConfig(
            strategy=strategy, severity=severity, clients=args.clients,
            seed=args.seed, val_fraction=args.val_fraction,
            min_samples_per_client=args.min_samples,
            test_strategy=args.test_strategy, max_samples=args.max_samples,
        )
        with_families = args.families or strategy == "family_skew"
        logger.info("building strategy=%s clients=%d seed=%d severity=%s",
                    strategy, args.clients, args.seed, severity)
        build_one(root, strategy, cfg, vectorized_dir, indices_path,
                  n_train_rows, with_families)


if __name__ == "__main__":
    main()