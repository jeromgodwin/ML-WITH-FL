"""Centralized trainer tests: small synthetic data, checkpointing, early stop."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedshield.config import TrainConfig  # noqa: E402
from src.federated.models.mlp import MLPConfig, build_mlp  # noqa: E402
from src.federated.training.centralized import (  # noqa: E402
    CHECKPOINT_FILENAME, TRAIN_REPORT_FILENAME, train_centralized,
)


@pytest.fixture()
def synthetic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 32)).astype(np.float32)
    # separable-ish pattern so training actually improves
    X[:, 0] += 2.0 * (rng.integers(0, 2, size=600) - 0.5)
    y = (X[:, 0] > 0).astype(np.int64)
    X[:300] *= 0.1
    X[300:] *= 0.1
    y = (X[:, 1] + X[:, 2] > 0).astype(np.int64)
    X = X + rng.normal(size=X.shape).astype(np.float32) * 0.1
    return X, y


def test_train_centralized_runs_and_checkpoints(synthetic, tmp_path):
    X, y = synthetic
    cfg = MLPConfig(input_dim=32, hidden_layers=(24, 12), dropout=0.1)
    tcfg = TrainConfig(batch_size=64, epochs=5, early_stopping_patience=0, seed=1)
    result = train_centralized(cfg, tcfg, X, y, X[:100], y[:100],
                               output_dir=tmp_path / "out")
    assert result.best_epoch >= 1
    assert result.epochs_run == 5
    assert result.params > 0
    assert result.train_time_s >= 0
    assert (tmp_path / "out" / CHECKPOINT_FILENAME).exists()
    report = json.loads((tmp_path / "out" / TRAIN_REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["model_config"]["input_dim"] == 32
    assert "metrics_val" in report
    checkpoint = torch.load(tmp_path / "out" / CHECKPOINT_FILENAME, weights_only=True)
    assert len(checkpoint) > 0


def test_train_centralized_reproducible(synthetic, tmp_path):
    X, y = synthetic
    cfg = MLPConfig(input_dim=32, hidden_layers=(16,), dropout=0.0)
    tcfg = TrainConfig(batch_size=64, epochs=2, early_stopping_patience=0, seed=7)
    r1 = train_centralized(cfg, tcfg, X, y, X[:100], y[:100])
    r2 = train_centralized(cfg, tcfg, X, y, X[:100], y[:100])
    assert r1.metrics_val["roc_auc"] == r2.metrics_val["roc_auc"]
    assert [h["train_loss"] for h in r1.history] == [h["train_loss"] for h in r2.history]


def test_train_centralized_early_stopping(synthetic, tmp_path):
    X, y = synthetic
    cfg = MLPConfig(input_dim=32, hidden_layers=(16,), dropout=0.0)
    tcfg = TrainConfig(batch_size=64, epochs=50, early_stopping_patience=2, seed=1)
    result = train_centralized(cfg, tcfg, X, y, X[:100], y[:100])
    assert result.epochs_run < 50
    assert result.best_epoch < result.epochs_run


def test_train_centralized_loss_decreases(synthetic, tmp_path):
    X, y = synthetic
    cfg = MLPConfig(input_dim=32, hidden_layers=(32, 16), dropout=0.0)
    tcfg = TrainConfig(batch_size=64, epochs=3, early_stopping_patience=0, seed=3)
    result = train_centralized(cfg, tcfg, X, y, X[:100], y[:100])
    losses = [h["train_loss"] for h in result.history]
    assert losses[-1] < losses[0]