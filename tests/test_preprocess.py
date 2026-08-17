"""Preprocess tests: fit-on-train-only scaler, artifact persistence, correctness."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.federated.data.preprocess import (  # noqa: E402
    MANIFEST_FILENAME, PreprocessManifest, apply_scaler, fit_scaler,
)


@pytest.fixture()
def X():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2381)).astype(np.float32)
    X[:, ::50] += 100.0  # make some features large-scale
    return X


def test_fit_scaler_reproducible_and_applied(X, tmp_path):
    m1 = fit_scaler(X, tmp_path / "a", seed=1, train_idx=np.arange(100))
    m2 = fit_scaler(X, tmp_path / "b", seed=1, train_idx=np.arange(100))
    assert m1.params_hash == m2.params_hash

    scaler = joblib.load(tmp_path / "a" / "scaler.joblib")
    scaled = apply_scaler(X[:100], scaler)
    # variance of scaled train rows must be ~1 per feature
    vars = scaled.var(axis=0)
    assert np.allclose(vars, 1.0, atol=1e-3)


def test_fit_only_on_train_rows(X, tmp_path):
    # same train rows, different (leaked) test rows -> params must be identical
    m1 = fit_scaler(X, tmp_path / "a", seed=1, train_idx=np.arange(0, 100))
    X2 = X.copy()
    X2[150:] *= 1000  # perturb rows outside the fit window
    m2 = fit_scaler(X2, tmp_path / "b", seed=1, train_idx=np.arange(0, 100))
    assert m1.params_hash == m2.params_hash


def test_manifest_persisted(X, tmp_path):
    m = fit_scaler(X, tmp_path / "a", seed=42, train_idx=np.arange(50))
    assert isinstance(m, PreprocessManifest)
    manifest_path = tmp_path / "a" / MANIFEST_FILENAME
    assert manifest_path.exists()
    import json
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["n_train_rows"] == 50
    assert data["seed"] == 42
    assert data["n_features"] == 2381


def test_apply_scaler_preserves_dtype(X, tmp_path):
    m = fit_scaler(X, tmp_path / "a", seed=1, train_idx=np.arange(50))
    scaler = joblib.load(tmp_path / "a" / "scaler.joblib")
    out = apply_scaler(X[:10], scaler)
    assert out.dtype == np.float32