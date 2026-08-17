"""Metric tests: correctness against hand-computed values."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.federated.evaluation.metrics import (  # noqa: E402
    compute_metrics, evaluate_model, predict_proba_chunked,
)


def test_compute_metrics_hand_values():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 1])
    p = np.array([0.9, 0.8, 0.2, 0.4, 0.6, 0.1, 0.7, 0.3])
    m = compute_metrics(y, p, threshold=0.5)
    # tp=4 (idx 0,1,4,6), fp=0, tn=3 (idx 2,3,5), fn=1 (idx 7); one inverted
    # pair (p=0.3 malicious < p=0.4 benign) -> auc matches sklearn's value
    assert m["confusion_matrix"] == {"tn": 3, "fp": 0, "fn": 1, "tp": 4}
    assert m["accuracy"] == pytest.approx(7 / 8)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(4 / 5)
    assert m["f1"] == pytest.approx(2 * (1.0 * 0.8) / (1.0 + 0.8))
    assert m["roc_auc"] == pytest.approx(0.9333333333333333)
    assert m["n_samples"] == 8


def test_compute_metrics_single_class_auc_none():
    m = compute_metrics(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]))
    assert m["roc_auc"] is None


def test_compute_metrics_empty_pred_class():
    m = compute_metrics(np.array([0, 0]), np.array([0.1, 0.2]))
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_predict_proba_chunked_matches_full():
    from src.federated.models.mlp import build_mlp, MLPConfig
    model = build_mlp(MLPConfig(input_dim=8, hidden_layers=(8,), dropout=0.0))
    model.eval()
    X = np.random.default_rng(0).normal(size=(500, 8)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 2, size=500)
    full = predict_proba_chunked(model, X, chunk=10_000)
    chunked = predict_proba_chunked(model, X, chunk=64)
    assert np.allclose(full, chunked, atol=1e-6)
    m = evaluate_model(model, X, y, chunk=64)
    assert set(["accuracy", "precision", "recall", "f1", "roc_auc"]) <= set(m.keys())