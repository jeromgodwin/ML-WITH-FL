"""Train and evaluate the centralized baseline on EMBER 2018_2.

Pipeline:
  1. load vectorized train/val/test splits (memmap) + split indices + scaler
  2. materialize labeled train/val arrays in RAM (scaled with the saved scaler)
  3. train MLP with early stopping, checkpoint best val model
  4. evaluate on the official test split (chunked, low RAM)
  5. export a self-contained model bundle + register it

Usage:
    python scripts/train_centralized.py [--data-dir data] [--epochs 20]
        [--batch-size 512] [--lr 1e-3] [--hidden 256,128] [--dropout 0.2]
        [--seed 42] [--version mlp-central-v1] [--skip-registry]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fedshield.config import TrainConfig  # noqa: E402
from fedshield.logging_setup import get_logger, log_event  # noqa: E402
from src.federated.data.dataset import load_split  # noqa: E402
from src.federated.data.preprocess import SCALER_FILENAME, MANIFEST_FILENAME  # noqa: E402
from src.federated.data.split import load_split_indices  # noqa: E402
from src.federated.evaluation.metrics import (  # noqa: E402
    compute_metrics, predict_proba_chunked,
)
from src.federated.model_bundle import export_bundle, register_bundle_in_registry  # noqa: E402
from src.federated.model_registry import ModelRegistry  # noqa: E402
from src.federated.models.mlp import MLPConfig, build_mlp  # noqa: E402
from src.federated.training.centralized import train_centralized  # noqa: E402

logger = get_logger(__name__)


def _materialize(vectorized_dir: Path, indices_path: Path, artifacts_dir: Path):
    """Load labeled train/val arrays (scaled) into RAM; test stays on disk.

    Gathered in chunks and the memmap mapping is dropped between gathers so
    Windows file-cache pages can be reclaimed before the next allocation.
    """
    import gc

    import joblib

    X_train, y_train = load_split(vectorized_dir, "train")
    X_test, _ = load_split(vectorized_dir, "test")
    indices = load_split_indices(indices_path)

    scaler_path = artifacts_dir / SCALER_FILENAME
    if not scaler_path.exists():
        raise FileNotFoundError(f"scaler not found: {scaler_path} (run scripts/prepare_ember.py)")
    scaler = joblib.load(scaler_path)
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
    labeled_train = train_idx[(y_train[train_idx] == 0) | (y_train[train_idx] == 1)]
    labeled_val = val_idx[(y_train[val_idx] == 0) | (y_train[val_idx] == 1)]

    Xtr = gather(X_train, labeled_train)
    del X_train
    gc.collect()
    Xva = gather(np.load(vectorized_dir / "X_train.npy", mmap_mode="r"), labeled_val)

    ytr = np.asarray(y_train[labeled_train], dtype=np.int8)
    yva = np.asarray(y_train[labeled_val], dtype=np.int8)
    return Xtr, ytr, Xva, yva, X_test, scale_inv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train centralized EMBER baseline")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden", type=str, default=None, help="comma list, e.g. 256,128")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--optimizer", default=None, choices=["adam", "adamw", "sgd"])
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", default="mlp-central-v1")
    parser.add_argument("--skip-registry", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_dir)
    vectorized_dir = root / "ember_2018_2" / "vectorized"
    artifacts_dir = root / "ember_2018_2" / "artifacts"
    indices_path = artifacts_dir / "split_indices.npz"
    out_dir = root / "ember_2018_2" / "models" / args.version

    hidden = tuple(int(h) for h in args.hidden.split(",")) if args.hidden else (256, 128)
    model_cfg = MLPConfig(input_dim=2381, hidden_layers=hidden,
                          dropout=args.dropout if args.dropout is not None else 0.2)

    train_cfg = TrainConfig(
        batch_size=args.batch_size or 512,
        learning_rate=args.lr if args.lr is not None else 1e-3,
        optimizer=args.optimizer or "adam",
        seed=args.seed,
        epochs=args.epochs or 20,
        early_stopping_patience=args.patience if args.patience is not None else 3,
    )

    log_event(logger, logging.INFO, "centralized_start",
              version=args.version, **model_cfg.to_dict(), **train_cfg.__dict__)

    Xtr, ytr, Xva, yva, X_test_mm, scale_inv = _materialize(vectorized_dir, indices_path, artifacts_dir)
    logger.info("arrays: train=%s val=%s test-on-disk=%s",
                Xtr.shape, Xva.shape, X_test_mm.shape)

    result = train_centralized(
        model_cfg, train_cfg, Xtr, ytr, Xva, yva,
        output_dir=out_dir, device="cpu",
    )

    model = build_mlp(model_cfg)
    model.load_state_dict(torch.load(out_dir / "best_model.pt", weights_only=True))

    y_test = np.load(vectorized_dir / "y_test.npy")
    y_test_prob = predict_proba_chunked(model, X_test_mm, chunk=20_000, scale_inv=scale_inv)
    test_metrics = compute_metrics(y_test, y_test_prob)
    log_event(logger, logging.INFO, "centralized_test", version=args.version, **test_metrics)

    # Export self-contained bundle
    scaler_path = artifacts_dir / SCALER_FILENAME
    bundle_dir = export_bundle(
        model, model_cfg, scaler_path,
        metrics={**result.to_dict(), "test": test_metrics},
        version=args.version,
        output_dir=out_dir / "bundle",
        extra_metadata={
            "train_time_s": round(result.train_time_s, 2),
            "best_epoch": result.best_epoch,
            "n_train": int(Xtr.shape[0]),
        },
    )

    if not args.skip_registry:
        registry = ModelRegistry(root / "ember_2018_2" / "registry")
        register_bundle_in_registry(bundle_dir, registry, args.version,
                                    algorithm="centralized")
        logger.info("registered %s in %s", args.version, registry.artifacts_dir)

    report = {
        "version": args.version,
        "model_config": model_cfg.to_dict(),
        "train_config": train_cfg.__dict__,
        "train_time_s": round(result.train_time_s, 2),
        "params": result.params,
        "model_size_bytes": result.model_size_bytes,
        "val": result.metrics_val,
        "test": test_metrics,
        "bundle_dir": str(bundle_dir),
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("report: %s", report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()