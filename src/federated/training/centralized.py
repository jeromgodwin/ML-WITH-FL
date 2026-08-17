"""Centralized baseline training: loop, validation, checkpointing, early stopping.

The trainer consumes already-scaled numpy arrays (scaling happens in the
preprocessing stage, fit on train only). It is the reference point against
which federated algorithms will be compared.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedshield.config import ModelConfig, TrainConfig
from fedshield.logging_setup import get_logger, log_event
from src.federated.evaluation.metrics import compute_metrics, predict_proba_chunked
from src.federated.models.mlp import MLPConfig, build_mlp, count_parameters, model_size_bytes
from src.utils.reproducibility import set_all_seeds

logger = get_logger(__name__)

CHECKPOINT_FILENAME = "best_model.pt"
TRAIN_REPORT_FILENAME = "training_report.json"


@dataclass
class CentralizedResult:
    """Everything produced by one centralized training run."""

    metrics_train: Dict[str, Any]
    metrics_val: Dict[str, Any]
    best_epoch: int
    epochs_run: int
    train_time_s: float
    params: int
    model_size_bytes: int
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics_train": self.metrics_train,
            "metrics_val": self.metrics_val,
            "best_epoch": self.best_epoch,
            "epochs_run": self.epochs_run,
            "train_time_s": round(self.train_time_s, 2),
            "params": self.params,
            "model_size_bytes": self.model_size_bytes,
            "history": self.history,
        }


def _make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    name = cfg.optimizer.lower()
    lr = cfg.learning_rate
    wd = cfg.weight_decay
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"unsupported optimizer: {cfg.optimizer}")


@torch.no_grad()
def _evaluate_val(model: nn.Module, X_val: np.ndarray, y_val: np.ndarray, batch: int) -> Dict[str, Any]:
    model.eval()
    y_prob = predict_proba_chunked(model, X_val, chunk=min(batch * 8, 100_000))
    return compute_metrics(y_val, y_prob)


def train_centralized(
    model_cfg: MLPConfig,
    train_cfg: TrainConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: Optional[Path] = None,
    device: str = "cpu",
) -> CentralizedResult:
    """Train the centralized baseline.

    Arrays are expected pre-scaled (fit on train only). Labels must be 0/1.
    Checkpoints the best validation model to output_dir/best_model.pt and
    writes training_report.json when output_dir is given.
    """
    set_all_seeds(train_cfg.seed)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)

    model = build_mlp(model_cfg)
    optimizer = _make_optimizer(model, train_cfg)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=0, drop_last=False,
    )

    patience = train_cfg.early_stopping_patience
    best_auc = -1.0
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).ravel()
            loss = criterion(logits, yb.float())
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        val_metrics = _evaluate_val(model, X_val, y_val, train_cfg.batch_size)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(n_batches, 1), 6),
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        history.append(epoch_metrics)
        log_event(logger, logging.INFO, f"centralized epoch {epoch}", **epoch_metrics)

        if val_metrics["roc_auc"] is not None and val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if patience > 0 and epoch - best_epoch >= patience and best_epoch > 0:
            logger.info("early stopping at epoch %d (best epoch %d, val auc %.4f)",
                        epoch, best_epoch, best_auc)
            break

    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics_train = compute_metrics(y_train, predict_proba_chunked(model, X_train, chunk=100_000))
    metrics_val = compute_metrics(y_val, predict_proba_chunked(model, X_val, chunk=100_000))

    result = CentralizedResult(
        metrics_train=metrics_train,
        metrics_val=metrics_val,
        best_epoch=best_epoch,
        epochs_run=len(history),
        train_time_s=train_time,
        params=count_parameters(model),
        model_size_bytes=model_size_bytes(model),
        history=history,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / CHECKPOINT_FILENAME)
        with open(output_dir / TRAIN_REPORT_FILENAME, "w", encoding="utf-8") as f:
            json.dump({
                "model_config": model_cfg.to_dict(),
                "train_config": _train_config_dict(train_cfg),
                **result.to_dict(),
            }, f, indent=2)
        logger.info("checkpoint saved: %s", output_dir / CHECKPOINT_FILENAME)

    return result


def _train_config_dict(cfg: TrainConfig) -> Dict[str, Any]:
    return {
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "optimizer": cfg.optimizer,
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "early_stopping_patience": cfg.early_stopping_patience,
    }