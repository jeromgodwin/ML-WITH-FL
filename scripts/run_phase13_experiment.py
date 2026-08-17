"""Phase 13 experiment: temporal drift detection and adaptive retraining.

Dataset: EMBER 2018_2 with genuine PE compile timestamps (header_timestamp, idx 626).
Split: median timestamp → era-0 (older, ~440k) and era-1 (newer, ~360k).
Era-1 is streamed in 5 sequential windows by timestamp.

Compares three strategies on the SAME official test set:
  A) Static: train once on era-0 only
  B) Periodic FL: retrain every 2 windows on cumulative data
  C) Drift-triggered FL: PSI detector triggers retraining on DRIFT_DETECTED + safety OK

Metrics (all on the SAME official EMBER test set):
  - final F1 (test set)
  - total communication bytes (sum over all FL runs)
  - total training time (wall seconds)
  - number of retraining events
  - time to recovery (wall seconds from trigger to model ready) for C
"""

from __future__ import annotations

import json
import os
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
from src.drift import DriftConfig, DriftDetector  # noqa: E402
from src.drift.safety import RetrainingSafety  # noqa: E402
from src.federated.data.dataset import load_split  # noqa: E402
from src.federated.fl.server import run_fl_experiment  # noqa: E402

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("configs/default.yaml")
DEFAULT_VECTORIZED = Path("data/ember_2018_2/vectorized")
DEFAULT_SCALER = Path("data/ember_2018_2/artifacts/scaler.joblib")
PARTITION_ROOT = Path("data")
OUTPUT_ROOT = Path("data/fl/phase13")

TS_FEATURE_IDX = 626  # header_timestamp
VALID_TS_MIN = 946684800   # 2000-01-01
VALID_TS_MAX = 1609459200  # 2021-01-01
N_WINDOWS = 5
RETRAIN_EVERY = 2


def load_data_and_split() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, float
]:
    """Load data and split into era-0 / era-1 by timestamp median."""
    X_train, y_train = load_split(DEFAULT_VECTORIZED, "train")
    X_test, y_test = load_split(DEFAULT_VECTORIZED, "test")
    ts = X_train[:, TS_FEATURE_IDX]
    valid = np.isfinite(ts) & (ts >= VALID_TS_MIN) & (ts <= VALID_TS_MAX)
    ts_valid = ts[valid].astype(np.int64)
    median_ts = float(np.median(ts_valid))
    logger.info("timestamp median: %s (era-0=older, era-1=newer)",
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(median_ts)))
    era0_mask = valid & (ts <= median_ts)
    era1_mask = valid & (ts > median_ts)
    X_era0, y_era0 = X_train[era0_mask], y_train[era0_mask]
    X_era1, y_era1 = X_train[era1_mask], y_train[era1_mask]
    logger.info("era-0: %d samples, era-1: %d samples", len(X_era0), len(X_era1))
    return X_era0, y_era0, X_era1, y_era1, X_test, y_test, ts, median_ts


def create_filtered_partition(
    base_partition_dir: Path,
    output_dir: Path,
    keep_pool_indices: np.ndarray,
) -> Path:
    """Create a new partition keeping only samples whose pool index is in keep_pool_indices."""
    output_dir.mkdir(parents=True, exist_ok=True)
    part_data = np.load(base_partition_dir / "client_indices.npz")
    pool = part_data["pool"]  # global indices into X_train
    keep_set = set(int(x) for x in keep_pool_indices)
    new_pool = np.array([p for p in pool if p in keep_set], dtype=np.int64)
    new_part = {"pool": new_pool}
    for key in part_data.keys():
        if key == "pool":
            continue
        client_pool = part_data[key]
        new_part[key] = np.array([p for p in client_pool if p in keep_set], dtype=np.int64)
    np.savez_compressed(output_dir / "client_indices.npz", **new_part)
    shutil.copy2(base_partition_dir / "manifest.json", output_dir / "manifest.json")
    shutil.copy2(base_partition_dir / "report.json", output_dir / "report.json")
    logger.info("created filtered partition at %s (pool=%d)",
                output_dir, len(new_pool))
    return output_dir


def run_fl_on_partition(
    cfg: ExperimentConfig,
    partition_dir: Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scale_inv: np.ndarray,
    output_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    """Run FL experiment on a partition and return key metrics."""
    X_train, y_train = load_split(DEFAULT_VECTORIZED, "train")
    res = run_fl_experiment(
        cfg, partition_dir, X_train, y_train, X_test, y_test,
        scale_inv=scale_inv, output_dir=output_dir, seed=seed
    )
    return {
        "final_global_f1": (res["final_global_test_metrics"] or {}).get("f1"),
        "final_global_auc": (res["final_global_test_metrics"] or {}).get("roc_auc"),
        "total_bytes_exchanged": res["communication"]["totals"]["total_bytes_exchanged"],
        "training_time_s": res["training_time_s"],
    }


def main() -> None:
    cfg = ExperimentConfig.from_yaml(DEFAULT_CONFIG)
    cfg.fl.num_rounds = 5
    cfg.fl.algorithm = "fedavg"
    cfg.fl.num_clients = 10

    # Load scaler
    scaler = joblib.load(DEFAULT_SCALER)
    scale_inv = np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float32)
    scale_inv = (1.0 / scale_inv).astype(np.float32)

    # Temporal split on full training data
    X_train, y_train = load_split(DEFAULT_VECTORIZED, "train")
    X_test, y_test = load_split(DEFAULT_VECTORIZED, "test")
    ts = X_train[:, TS_FEATURE_IDX]
    valid = np.isfinite(ts) & (ts >= VALID_TS_MIN) & (ts <= VALID_TS_MAX)
    ts_valid = ts[valid].astype(np.int64)
    median_ts = float(np.median(ts_valid))
    logger.info("timestamp median: %s",
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(median_ts)))

    era0_mask = valid & (ts <= median_ts)
    era1_mask = valid & (ts > median_ts)

    # Base iid partition
    base_partition = PARTITION_ROOT / f"iid-c{cfg.fl.num_clients}-s{cfg.seed}"
    part_data = np.load(base_partition / "client_indices.npz")
    pool = part_data["pool"]  # global indices into X_train

    # Map pool to eras
    ts_pool = ts[pool]
    era0_pool_mask = (ts_pool <= median_ts) & (ts_pool >= VALID_TS_MIN) & (ts_pool <= VALID_TS_MAX)
    era1_pool_mask = (ts_pool > median_ts) & (ts_pool <= VALID_TS_MAX)
    era0_pool = pool[era0_pool_mask]
    era1_pool = pool[era1_pool_mask]
    logger.info("pool split: era0=%d, era1=%d", len(era0_pool), len(era1_pool))

    # Split era-1 pool into sequential windows by timestamp
    ts_era1_pool = ts_pool[era1_pool_mask]
    era1_order = np.argsort(ts_era1_pool)
    n_per = len(era1_pool) // N_WINDOWS
    era1_window_pools = []
    for i in range(N_WINDOWS):
        start = i * n_per
        end = start + n_per if i < N_WINDOWS - 1 else len(era1_pool)
        era1_window_pools.append(era1_pool[era1_order[start:end]])

    results = {"A": {}, "B": {}, "C": {}}

    # ============================================================
    # A) STATIC MODEL (era-0 only)
    # ============================================================
    logger.info("=== A) Static model (era-0 only) ===")
    part_A = OUTPUT_ROOT / "partition_A_static"
    create_filtered_partition(base_partition, part_A, era0_pool)
    out_A = OUTPUT_ROOT / "A_static"
    res_A = run_fl_on_partition(cfg, part_A, X_test, y_test, scale_inv, out_A, cfg.seed)
    results["A"] = {
        "strategy": "static_era0_only",
        "final_test_f1": res_A["final_global_f1"],
        "total_bytes": res_A["total_bytes_exchanged"],
        "training_time_s": res_A["training_time_s"],
        "retrains": 0,
    }
    logger.info("Static F1: %.4f", res_A["final_global_f1"])

    # ============================================================
    # B) PERIODIC FL (retrain every RETRAIN_EVERY windows)
    # ============================================================
    logger.info("=== B) Periodic FL (every %d windows) ===")
    cumulative_pool = np.concatenate([era0_pool, era1_window_pools[0]])
    B_bytes = 0
    B_time = 0.0
    B_retrains = 0
    last_B_f1 = 0.0

    for w_idx in range(N_WINDOWS):
        if w_idx > 0:
            cumulative_pool = np.concatenate([cumulative_pool, era1_window_pools[w_idx]])
        # Retrain periodically
        if (w_idx + 1) % RETRAIN_EVERY == 0 or w_idx == N_WINDOWS - 1:
            logger.info("  Periodic retrain at window %d (cumulative pool: %d)", w_idx, len(cumulative_pool))
            part_B = OUTPUT_ROOT / f"partition_B_w{w_idx}"
            create_filtered_partition(base_partition, part_B, cumulative_pool)
            out_B = OUTPUT_ROOT / f"B_periodic_w{w_idx}"
            res_B = run_fl_on_partition(cfg, part_B, X_test, y_test, scale_inv, out_B, cfg.seed)
            B_bytes += res_B["total_bytes_exchanged"]
            B_time += res_B["training_time_s"]
            B_retrains += 1
            last_B_f1 = res_B["final_global_f1"]

    results["B"] = {
        "strategy": "periodic_fl",
        "final_test_f1": last_B_f1,
        "total_bytes": B_bytes,
        "training_time_s": B_time,
        "retrains": B_retrains,
    }
    logger.info("Periodic final F1: %.4f", last_B_f1)

    # ============================================================
    # C) DRIFT-TRIGGERED FL
    # ============================================================
    logger.info("=== C) Drift-triggered FL ===")
    drift_cfg = DriftConfig(
        enabled=True,
        reference_frac=0.5,
        psi_suspect_threshold=0.1,
        psi_detected_threshold=0.2,
        min_new_samples=5000,
        cooldown_hours=0.5,   # short for experiment
        max_frequency_per_day=10,
    )
    detector = DriftDetector(drift_cfg, X_train[era0_mask])
    safety = RetrainingSafety(drift_cfg)

    cumulative_pool = era0_pool.copy()
    C_bytes = 0
    C_time = 0.0
    C_retrains = 0
    retrain_events = []
    last_C_f1 = results["A"]["final_test_f1"]

    for w_idx in range(N_WINDOWS):
        cumulative_pool = np.concatenate([cumulative_pool, era1_window_pools[w_idx]])
        X_w = X_train[era1_window_pools[w_idx]]
        drift_result = detector.compute(X_w)
        logger.info("  Window %d: PSI=%.4f status=%s", w_idx, drift_result.psi, drift_result.status)

        if drift_result.status == "DRIFT_DETECTED":
            safety_check = safety.check(len(era1_window_pools[w_idx]))
            if safety_check.allowed:
                logger.info("    -> TRIGGER: drift detected + safety OK, retraining...")
                part_C = OUTPUT_ROOT / f"partition_C_w{w_idx}"
                create_filtered_partition(base_partition, part_C, cumulative_pool)
                out_C = OUTPUT_ROOT / f"C_drift_w{w_idx}"
                t0 = time.time()
                res_C = run_fl_on_partition(cfg, part_C, X_test, y_test, scale_inv, out_C, cfg.seed)
                C_bytes += res_C["total_bytes_exchanged"]
                C_time += res_C["training_time_s"]
                C_retrains += 1
                last_C_f1 = res_C["final_global_f1"]
                retrain_events.append({
                    "window": w_idx,
                    "psi": drift_result.psi,
                    "retrain_time_s": time.time() - t0,
                    "f1_after": res_C["final_global_f1"],
                })
                safety.record_retrain(cfg.fl.num_rounds)
            else:
                logger.info("    -> BLOCKED by safety: %s", safety_check.reason)

    results["C"] = {
        "strategy": "drift_triggered_fl",
        "final_test_f1": last_C_f1,
        "total_bytes": C_bytes,
        "training_time_s": C_time,
        "retrains": C_retrains,
        "retrain_events": retrain_events,
    }
    if C_retrains > 0:
        logger.info("Drift-triggered final F1: %.4f", last_C_f1)
    else:
        logger.info("No drift triggered; falls back to static F1: %.4f", results["A"]["final_test_f1"])

    # Save comparison
    comparison = {
        "experiment": "phase13_temporal_drift",
        "split": "temporal (median timestamp)",
        "era0_samples": int(np.sum(era0_mask)),
        "era1_samples": int(np.sum(era1_mask)),
        "n_windows": N_WINDOWS,
        "median_timestamp": median_ts,
        "results": results,
    }
    (OUTPUT_ROOT / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()