"""Unified experiment engine (Phase 15).

Wraps existing centralized and FL runners without breaking endpoint/FL code.
Each run gets a unique experiment ID and never overwrites previous results.
Stores: config, environment, per-round, per-client, final, training time,
resource, communication, drift, security, privacy, model metadata, logs, plots.

Usage via scripts/run_experiment.py.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np

from fedshield.config import ExperimentConfig
from fedshield.logging_setup import get_logger
from src.federated.experiments.environment import (
    collect_environment_metadata,
    collect_reproducibility_record,
)
from src.federated.experiments.storage import ExperimentStorage, DEFAULT_EXPERIMENTS_ROOT
from src.federated.experiments.plots import plot_f1_per_round, plot_communication
from src.utils.reproducibility import generate_experiment_id

logger = get_logger(__name__)

DEFAULT_VECTORIZED = Path("data/ember_2018_2/vectorized")
DEFAULT_SCALER = Path("data/ember_2018_2/artifacts/scaler.joblib")


def _load_data(vectorized: Path = DEFAULT_VECTORIZED):
    from src.federated.data.dataset import load_split

    X_train, y_train = load_split(vectorized, "train")
    X_test, y_test = load_split(vectorized, "test")
    return X_train, y_train, X_test, y_test


def _load_scaler(scaler_path: Path = DEFAULT_SCALER):
    import joblib

    scaler = joblib.load(scaler_path)
    scale_inv = np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float32)
    scale_inv = (1.0 / scale_inv).astype(np.float32)
    return scale_inv


def run_unified_experiment(
    cfg: ExperimentConfig,
    raw_config: Optional[Dict[str, Any]] = None,
    root: Path | str = DEFAULT_EXPERIMENTS_ROOT,
    experiment_id: Optional[str] = None,
    vectorized: Path = DEFAULT_VECTORIZED,
    scaler_path: Path = DEFAULT_SCALER,
) -> Dict[str, Any]:
    """Run one unified experiment (centralized or federated).

    - Generates a unique experiment ID
    - Never overwrites existing results
    - Stores all required artifacts + plots
    - Returns the summary dict and storage path
    """
    # Generate unique ID if not provided
    storage = ExperimentStorage.create(cfg, root=root, experiment_id=experiment_id)
    exp_id = storage.experiment_id
    logger.info("starting unified experiment %s (algorithm=%s)", exp_id, cfg.fl.algorithm)

    t0 = time.perf_counter()
    # Environment + reproducibility (collect before run)
    env_meta = collect_environment_metadata()
    # Resolve partition dir if applicable
    partition_dir = None
    if cfg.fl.algorithm != "centralized":
        partition_dir = Path("data") / f"{cfg.partition.strategy}-c{cfg.fl.num_clients}-s{cfg.seed}"
        # Fallback to iid if missing
        if not partition_dir.exists():
            # also try without strategy prefix variations
            alt = Path("data") / f"iid-c{cfg.fl.num_clients}-s{cfg.seed}"
            if alt.exists():
                partition_dir = alt
    repro = collect_reproducibility_record(cfg, partition_dir=partition_dir, vectorized_dir=vectorized)

    # Save config / env / reproducibility early
    storage.save_config(cfg, raw_source=raw_config or cfg.to_dict())
    storage.save_environment(env_meta)
    storage.save_reproducibility(repro)
    storage.write_json("metadata.json", {"experiment_id": exp_id, "created_at": datetime.now(timezone.utc).isoformat(), "config_fingerprint": repro.get("config_fingerprint")})

    # Run the actual training
    results: Dict[str, Any] = {}
    error: Optional[str] = None
    try:
        if cfg.fl.algorithm == "centralized":
            results = _run_centralized(cfg, vectorized, storage)
        else:
            results = _run_federated(cfg, vectorized, scaler_path, storage)
    except Exception as e:
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        logger.error("experiment %s failed: %s", exp_id, error)
        results = {"error": error, "experiment": {"algorithm": cfg.fl.algorithm}}

    # Enrich results with storage-level metadata
    elapsed = time.perf_counter() - t0
    results.setdefault("experiment", {})["experiment_id"] = exp_id
    results.setdefault("experiment", {})["elapsed_wall_s"] = round(elapsed, 3)
    results["experiment_id"] = exp_id
    results["created_at"] = datetime.now(timezone.utc).isoformat()
    results["environment"] = env_meta
    results["reproducibility"] = repro
    if error:
        results["status"] = "failed"
        results["error"] = error
    else:
        results["status"] = "completed"

    # Standardize keys for aggregation (no invented values: keep None where missing)
    # per-round metrics
    per_round = results.get("rounds") or []
    # final metrics
    final = results.get("final_global_test_metrics") or results.get("final_metrics") or results.get("test") or {}
    # training time
    training_time_s = results.get("training_time_s") or results.get("train_time_s")
    # communication
    communication = results.get("communication")
    # resource
    resource = results.get("resource")
    # drift
    drift = results.get("drift")
    # security
    security = {}
    if "attack" in results or "defense" in results:
        security = {k: results[k] for k in ("attack", "defense") if k in results}
    # privacy
    privacy = results.get("privacy") or ({"enabled": cfg.privacy.enabled, "noise_multiplier": cfg.privacy.noise_multiplier, "max_grad_norm": cfg.privacy.max_grad_norm, "delta": cfg.privacy.delta} if cfg.privacy.enabled else None)
    # model metadata
    model_meta = results.get("model") or cfg.model.to_dict() if hasattr(cfg.model, "to_dict") else dict(cfg.model.__dict__)

    unified_summary: Dict[str, Any] = {
        "experiment_id": exp_id,
        "experiment": results.get("experiment", {}),
        "config": cfg.to_dict(),
        "environment": env_meta,
        "reproducibility": repro,
        "rounds": per_round,
        "per_client_metrics": results.get("per_client_metrics"),
        "final_global_test_metrics": final,
        "final_metrics": final,
        "training_time_s": training_time_s,
        "communication": communication,
        "resource": resource,
        "drift": drift,
        "security": security if security else None,
        "privacy": privacy,
        "model_metadata": model_meta,
        "logs": results.get("logs"),
        "status": results.get("status", "completed"),
    }
    # Also keep raw results for debugging
    unified_summary["_raw"] = {k: v for k, v in results.items() if k not in unified_summary}

    # Persist unified summary via storage
    storage.save_metrics(unified_summary)

    # Plots (best-effort, never fail the experiment)
    try:
        if per_round:
            plot_f1_per_round(per_round, storage.dir / "plots" / "f1_per_round.png", title=f"{exp_id} F1 per round")
            plot_communication(per_round, storage.dir / "plots" / "bytes_per_round.png", title=f"{exp_id} bytes per round")
    except Exception as e:
        logger.warning("plot generation failed for %s: %s", exp_id, e)

    # Also write aggregated convenience files in the experiment dir
    # (CSV/JSON comparison for single experiment is trivial but useful for scripts)
    try:
        storage.write_json("summary.json", unified_summary)
    except FileExistsError:
        pass

    logger.info("unified experiment %s completed (status=%s) in %.1fs -> %s", exp_id, unified_summary["status"], elapsed, storage.dir)
    return {"experiment_id": exp_id, "storage_dir": str(storage.dir), "summary": unified_summary}


def _run_centralized(cfg: ExperimentConfig, vectorized: Path, storage: ExperimentStorage) -> Dict[str, Any]:
    """Delegate to train_centralized without breaking its implementation."""
    from src.federated.data.dataset import load_split
    from src.federated.data.split import load_split_indices
    from src.federated.data.preprocess import SCALER_FILENAME, MANIFEST_FILENAME
    from src.federated.models.mlp import MLPConfig
    from src.federated.training.centralized import train_centralized
    import joblib
    import gc

    root = Path(cfg.data.data_dir)
    vectorized_dir = vectorized
    artifacts_dir = root / "ember_2018_2" / "artifacts"
    indices_path = artifacts_dir / "split_indices.npz"

    # materialize as in train_centralized.py but inline for unified storage
    X_train_mm, y_train_mm = load_split(vectorized_dir, "train")
    X_test_mm, _ = load_split(vectorized_dir, "test")
    indices = load_split_indices(indices_path)
    scaler = joblib.load(artifacts_dir / SCALER_FILENAME)
    scale = scaler.scale_.astype(np.float32)
    scale_inv = np.zeros_like(scale)
    scale_inv[scale != 0] = 1.0 / scale[scale != 0]

    def gather(X_src, idx, chunk=50_000):
        out = np.empty((len(idx), X_src.shape[1]), dtype=np.float32)
        for start in range(0, len(idx), chunk):
            block = np.asarray(X_src[idx[start:start + chunk]], dtype=np.float32)
            block *= scale_inv
            out[start:start + chunk] = block
            del block
        return out

    train_idx = indices["train_idx"]
    val_idx = indices["val_idx"]
    labeled_train = train_idx[(y_train_mm[train_idx] == 0) | (y_train_mm[train_idx] == 1)]
    labeled_val = val_idx[(y_train_mm[val_idx] == 0) | (y_train_mm[val_idx] == 1)]
    # Gather both splits before releasing memmap to avoid second mmap load
    Xtr = gather(X_train_mm, labeled_train)
    Xva = gather(X_train_mm, labeled_val)
    del X_train_mm
    gc.collect()
    ytr = np.asarray(y_train_mm[labeled_train], dtype=np.int8)
    yva = np.asarray(y_train_mm[labeled_val], dtype=np.int8)

    model_cfg = MLPConfig(
        input_dim=int(Xtr.shape[1]),
        hidden_layers=tuple(cfg.model.hidden_layers),
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
    )
    from fedshield.config import TrainConfig

    train_cfg = TrainConfig(
        batch_size=cfg.train.batch_size,
        local_epochs=cfg.train.local_epochs,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        optimizer=cfg.train.optimizer,
        seed=cfg.train.seed,
        epochs=cfg.train.epochs,
        early_stopping_patience=cfg.train.early_stopping_patience,
    )
    result = train_centralized(
        model_cfg, train_cfg, Xtr, ytr, Xva, yva, output_dir=storage.dir / "model", device="cpu"
    )
    # evaluate on test
    from src.federated.evaluation.metrics import compute_metrics, predict_proba_chunked
    from src.federated.models.mlp import build_mlp
    import torch

    model = build_mlp(model_cfg)
    model.load_state_dict(torch.load(storage.dir / "model" / "best_model.pt", weights_only=True))
    y_test = np.load(vectorized_dir / "y_test.npy")
    y_prob = predict_proba_chunked(model, X_test_mm, chunk=20000, scale_inv=scale_inv)
    test_metrics = compute_metrics(y_test, y_prob)

    # Convert to unified results shape
    history = result.history
    rounds = [
        {
            "round": h["epoch"],
            "global_eval": {"f1": h.get("val_f1"), "roc_auc": h.get("val_roc_auc"), "accuracy": h.get("val_accuracy")},
            "avg_client_f1": None,
            "worst_client_f1": None,
        }
        for h in history
    ]
    return {
        "experiment": {"algorithm": "centralized", "strategy": "centralized"},
        "rounds": rounds,
        "final_global_test_metrics": test_metrics,
        "final_metrics": test_metrics,
        "training_time_s": result.train_time_s,
        "communication": None,
        "resource": None,
        "drift": None,
        "privacy": {"enabled": cfg.privacy.enabled} if cfg.privacy.enabled else None,
        "model": model_cfg.to_dict(),
        "history": history,
    }


def _run_federated(
    cfg: ExperimentConfig,
    vectorized: Path,
    scaler_path: Path,
    storage: ExperimentStorage,
) -> Dict[str, Any]:
    """Delegate to run_fl_experiment without breaking its implementation."""
    from src.federated.data.dataset import load_split
    import joblib

    scale_inv = _load_scaler(scaler_path)
    X_train, y_train = load_split(vectorized, "train")
    X_test, y_test = load_split(vectorized, "test")
    partition_dir = Path("data") / f"{cfg.partition.strategy}-c{cfg.fl.num_clients}-s{cfg.seed}"
    if not partition_dir.exists():
        raise FileNotFoundError(f"partition not found: {partition_dir}")

    from src.federated.fl.server import run_fl_experiment

    # run_fl_experiment writes its own summary/config/rounds into output_dir;
    # we point it at storage.dir / 'fl_raw' then merge into unified summary.
    fl_raw_dir = storage.dir / "fl_raw"
    # Pass controller implicitly via cfg if resource enabled (handled inside run_fl_experiment)
    results = run_fl_experiment(
        cfg, partition_dir, X_train, y_train, X_test, y_test,
        scale_inv=scale_inv, output_dir=fl_raw_dir, seed=cfg.seed
    )
    # Privacy: run_fl_experiment now returns a full DP report (clipping+noise details + epsilon estimate)
    # Only attach a stub if the server did not already report privacy (backward compat for old runs)
    if cfg.privacy.enabled and "privacy" not in results:
        results["privacy"] = {
            "enabled": True,
            "noise_multiplier": cfg.privacy.noise_multiplier,
            "max_grad_norm": cfg.privacy.max_grad_norm,
            "delta": cfg.privacy.delta,
            "accounting_mode": cfg.privacy.accounting_mode,
            "note": "DP stub — server privacy report missing (should not happen after Phase 17)",
        }
    # drift metrics: if enabled, the adaptive retraining is out-of-band; for unified
    # we record the drift config and that no automatic drift round was triggered in FL
    if cfg.endpoint.drift.enabled:
        results["drift"] = {
            "enabled": True,
            "config": cfg.endpoint.drift.__dict__,
            "note": "drift detection is evaluated via separate temporal pipeline; FL run records config only",
        }
    return results
