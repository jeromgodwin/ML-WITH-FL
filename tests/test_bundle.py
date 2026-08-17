"""Bundle tests: export, cross-process load, inference parity."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.federated.models.mlp import MLPConfig, build_mlp  # noqa: E402
from src.federated.model_bundle import (  # noqa: E402
    BUNDLE_MANIFEST, InferenceBundle, export_bundle,
)


@pytest.fixture()
def trained_model():
    cfg = MLPConfig(input_dim=16, hidden_layers=(12,), dropout=0.0)
    model = build_mlp(cfg)
    model.eval()
    return model, cfg


def _dummy_scaler(tmp_path):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(with_mean=False)
    scaler.mean_ = None
    scaler.var_ = np.ones(16)
    scaler.scale_ = np.ones(16, dtype=np.float64)
    scaler.n_features_in_ = 16
    path = tmp_path / "scaler.joblib"
    import joblib
    joblib.dump(scaler, path)
    return path


def test_export_load_same_process(trained_model, tmp_path):
    model, cfg = trained_model
    scaler_path = _dummy_scaler(tmp_path)
    metrics = {"accuracy": 0.9, "roc_auc": 0.95}
    bundle_dir = export_bundle(model, cfg, scaler_path, metrics, "v-test", tmp_path / "b")

    manifest = json.loads((bundle_dir / BUNDLE_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["version"] == "v-test"
    assert manifest["input_dim"] == 16
    assert manifest["params"] > 0
    assert (bundle_dir / "feature_schema.json").exists()
    assert (bundle_dir / "scaler.joblib").exists()

    loaded = InferenceBundle.load(bundle_dir)
    X = np.random.default_rng(0).normal(size=(20, 16)).astype(np.float32)
    with torch.no_grad():
        expected = torch.sigmoid(model(torch.from_numpy(X))).numpy().ravel()
    assert np.allclose(loaded.predict_proba(X), expected, atol=1e-6)


def test_load_from_other_process(trained_model, tmp_path):
    """The bundle must be loadable by a fresh process (inference independence)."""
    model, cfg = trained_model
    scaler_path = _dummy_scaler(tmp_path)
    bundle_dir = export_bundle(model, cfg, scaler_path, {"accuracy": 0.9}, "v-xproc",
                               tmp_path / "b")

    probe = (
        "import sys, json, numpy as np; sys.path.insert(0, r'%s'); "
        "from src.federated.model_bundle import InferenceBundle; "
        "b = InferenceBundle.load(r'%s'); "
        "X = np.random.default_rng(3).normal(size=(8, 16)).astype(np.float32); "
        "print(json.dumps({'proba': b.predict_proba(X).tolist(), "
        "'n_features': b.config.input_dim, 'version': b.manifest['version']}))"
    ) % (str(Path(__file__).resolve().parents[1]), str(bundle_dir))
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout.strip().splitlines()[-1])
    assert data["n_features"] == 16
    assert data["version"] == "v-xproc"

    with torch.no_grad():
        expected = torch.sigmoid(model(torch.from_numpy(
            np.random.default_rng(3).normal(size=(8, 16)).astype(np.float32)))).numpy().ravel()
    assert np.allclose(data["proba"], expected, atol=1e-6)


def test_bundle_rejects_wrong_input_dim(trained_model, tmp_path):
    model, cfg = trained_model
    scaler_path = _dummy_scaler(tmp_path)
    bundle_dir = export_bundle(model, cfg, scaler_path, {}, "v-dim", tmp_path / "b")
    loaded = InferenceBundle.load(bundle_dir)
    with pytest.raises(ValueError):
        loaded.preprocess(np.zeros((2, 15), dtype=np.float32))