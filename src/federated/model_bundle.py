"""Self-contained model bundles: export and load for cross-process inference.

A bundle directory contains everything needed to run inference in another
process without access to the training environment:
    <version>/
        bundle.json          manifest (version, schema, scaler, config, metrics)
        model.pt             PyTorch state_dict
        model_config.json    MLPConfig
        scaler.joblib        fitted preprocessing pipeline
        feature_schema.json  2381 canonical feature names
        metrics.json         evaluation metrics
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import torch

from fedshield.logging_setup import get_logger
from src.federated.data.feature_schema import FEATURE_NAMES, FEATURE_VERSION, N_FEATURES
from src.federated.models.mlp import MLPConfig, build_mlp, count_parameters, model_size_bytes

logger = get_logger(__name__)

BUNDLE_MANIFEST = "bundle.json"
MODEL_FILE = "model.pt"
MODEL_CONFIG_FILE = "model_config.json"
SCALER_FILE = "scaler.joblib"
SCHEMA_FILE = "feature_schema.json"
METRICS_FILE = "metrics.json"


def export_bundle(
    model: torch.nn.Module,
    model_cfg: MLPConfig,
    scaler_path: Path,
    metrics: Dict[str, Any],
    version: str,
    output_dir: Path,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a self-contained bundle and return its directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / MODEL_FILE)
    (output_dir / MODEL_CONFIG_FILE).write_text(
        json.dumps(model_cfg.to_dict(), indent=2), encoding="utf-8")
    (output_dir / METRICS_FILE).write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / SCHEMA_FILE).write_text(
        json.dumps({
            "feature_version": FEATURE_VERSION,
            "n_features": N_FEATURES,
            "names": FEATURE_NAMES,
        }, indent=2), encoding="utf-8")

    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        joblib.dump(scaler, output_dir / SCALER_FILE)
    elif (output_dir / SCALER_FILE).exists():
        (output_dir / SCALER_FILE).unlink()

    manifest = {
        "version": version,
        "feature_version": FEATURE_VERSION,
        "n_features": N_FEATURES,
        "input_dim": model_cfg.input_dim,
        "params": count_parameters(model),
        "model_size_bytes": model_size_bytes(model),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_file": MODEL_FILE,
        "model_config_file": MODEL_CONFIG_FILE,
        "scaler_file": SCALER_FILE if (output_dir / SCALER_FILE).exists() else None,
        "schema_file": SCHEMA_FILE,
        "metrics_file": METRICS_FILE,
        **(extra_metadata or {}),
    }
    (output_dir / BUNDLE_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("bundle exported: %s (version %s, %d params)",
                output_dir, version, manifest["params"])
    return output_dir


@dataclass
class InferenceBundle:
    """Loadable inference object; model + pipeline + schema together."""

    model: torch.nn.Module
    config: MLPConfig
    scaler: Any
    schema: Dict[str, Any]
    manifest: Dict[str, Any]

    @classmethod
    def load(cls, bundle_dir: Path, device: str = "cpu") -> "InferenceBundle":
        bundle_dir = Path(bundle_dir)
        manifest = json.loads((bundle_dir / BUNDLE_MANIFEST).read_text(encoding="utf-8"))
        model_cfg = MLPConfig(**json.loads((bundle_dir / MODEL_CONFIG_FILE).read_text(encoding="utf-8")))
        model = build_mlp(model_cfg)
        state = torch.load(bundle_dir / MODEL_FILE, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        scaler = None
        if (bundle_dir / SCALER_FILE).exists():
            scaler = joblib.load(bundle_dir / SCALER_FILE)
        schema = json.loads((bundle_dir / SCHEMA_FILE).read_text(encoding="utf-8"))
        return cls(model=model, config=model_cfg, scaler=scaler, schema=schema, manifest=manifest)

    def preprocess(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.config.input_dim:
            raise ValueError(
                f"expected {self.config.input_dim} features, got {X.shape[1]}")
        if self.scaler is not None:
            scale = np.where(self.scaler.scale_ == 0, 1.0, self.scaler.scale_)
            X = X / scale.astype(np.float32)
        return X

    @torch.no_grad()
    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        X = self.preprocess(X)
        model = self.model
        out = np.empty(X.shape[0], dtype=np.float32)
        for start in range(0, X.shape[0], 4096):
            xb = torch.from_numpy(X[start:start + 4096])
            out[start:start + 4096] = model(xb).numpy().ravel()
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.predict_logits(X)))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def register_bundle_in_registry(
    bundle_dir: Path,
    registry: Any,
    version: str,
    algorithm: str = "centralized",
    training_round: Optional[int] = None,
):
    """Register an exported bundle with the ModelRegistry (approves it)."""
    from datetime import datetime, timezone

    from src.federated.model_registry import FeatureSchema, ModelMetadata

    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / BUNDLE_MANIFEST).read_text(encoding="utf-8"))
    metrics = json.loads((bundle_dir / METRICS_FILE).read_text(encoding="utf-8"))
    schema = json.loads((bundle_dir / SCHEMA_FILE).read_text(encoding="utf-8"))

    metadata = ModelMetadata(
        version=version,
        algorithm=algorithm,
        training_round=training_round,
        feature_schema=FeatureSchema(
            feature_names=tuple(schema["names"]),
            feature_types=("float32",) * schema["n_features"],
            preprocessing_version=f"ember_v{manifest.get('feature_version')}_std",
            created_at=datetime.now(timezone.utc).isoformat(),
            model_version=version,
        ),
        metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        created_at=datetime.now(timezone.utc).isoformat(),
        num_parameters=int(manifest.get("params", 0)),
        input_dim=int(manifest.get("n_features", 0)),
    )
    entry = registry.register(
        metadata=metadata,
        artifact_source=bundle_dir / MODEL_FILE,
        expected_input_dim=metadata.input_dim,
        copy_artifact=True,
    )
    registry.approve(version, metrics=metadata.metrics)
    return entry