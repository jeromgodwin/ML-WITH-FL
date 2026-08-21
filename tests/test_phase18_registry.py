"""Phase 18 tests — Model registry and automatic model deployment.

Flow: FL training → candidate model → validation → registry → approved model → client update → endpoint uses new model
"""

import json
import pathlib
import torch
from pathlib import Path

from src.federated.model_registry import ModelRegistry
from src.federated.registry.deployment import fl_checkpoint_to_registry, endpoint_discover_and_update
from src.federated.registry.updater import EndpointModelUpdater
from src.interfaces import FeatureSchema, ModelMetadata
from src.federated.data.feature_schema import FEATURE_NAMES, FEATURE_VERSION, N_FEATURES


def _make_schema(preprocessing_version="ember_v17_std", name="v1") -> FeatureSchema:
    return FeatureSchema(
        feature_names=tuple(FEATURE_NAMES),
        feature_types=("float32",) * N_FEATURES,
        preprocessing_version=preprocessing_version,
        created_at="2026-01-01T00:00:00",
        model_version=name,
    )


def _make_metadata(version: str, schema: FeatureSchema, input_dim: int = 2381, algo: str = "fedavg") -> ModelMetadata:
    return ModelMetadata(
        version=version,
        algorithm=algo,
        training_round=5,
        feature_schema=schema,
        metrics={"accuracy": 0.92, "f1": 0.91},
        created_at="2026-01-01T00:00:00",
        num_parameters=642817,
        input_dim=input_dim,
    )


def _dummy_checkpoint(tmp_path: Path, name: str = "ckpt.pt") -> Path:
    # Minimal valid state_dict for the MLP (256,128) - just enough to load
    # Build a real model and save it
    from src.federated.models.mlp import MLPConfig, build_mlp
    cfg = MLPConfig(input_dim=2381, hidden_layers=(256, 128))
    model = build_mlp(cfg)
    p = tmp_path / name
    torch.save(model.state_dict(), p)
    return p


def test_candidate_validation_success_activation(tmp_path):
    """Candidate → validation success → activation (ACTIVE, previous ARCHIVED)."""
    ckpt1 = _dummy_checkpoint(tmp_path, "ckpt1.pt")
    ckpt2 = _dummy_checkpoint(tmp_path, "ckpt2.pt")
    schema = _make_schema()
    registry = ModelRegistry(tmp_path / "registry")

    # First model: candidate → validated → active
    e1 = fl_checkpoint_to_registry(
        checkpoint_path=ckpt1,
        model_cfg=__import__("src.federated.models.mlp", fromlist=["MLPConfig"]).MLPConfig(input_dim=2381, hidden_layers=(256, 128)),
        scaler_path=Path("nonexistent"),
        metrics={"accuracy": 0.91},
        version="fl-v1",
        algorithm="fedavg",
        training_round=5,
        feature_schema=schema,
        preprocessing_version=schema.preprocessing_version,
        configuration={"model": {"input_dim": 2381}},
        registry=registry,
        expected_input_dim=2381,
        validation_metrics={"accuracy": 0.91, "f1": 0.90},
    )
    assert e1.status == "ACTIVE"
    assert e1.version == "fl-v1"
    assert registry.get_active().version == "fl-v1"
    # Track required fields
    d = e1.to_dict()
    for k in ("model_id", "version", "algorithm", "training_round", "feature_schema_version", "preprocessing_version", "configuration", "validation_metrics", "timestamp", "status"):
        assert k in d, f"missing {k}"
    assert d["status"] == "ACTIVE"

    # Second model: should archive previous
    e2 = fl_checkpoint_to_registry(
        checkpoint_path=ckpt2,
        model_cfg=__import__("src.federated.models.mlp", fromlist=["MLPConfig"]).MLPConfig(input_dim=2381, hidden_layers=(256, 128)),
        scaler_path=Path("nonexistent"),
        metrics={"accuracy": 0.93},
        version="fl-v2",
        algorithm="fedavg",
        training_round=10,
        feature_schema=schema,
        preprocessing_version=schema.preprocessing_version,
        configuration={"model": {"input_dim": 2381}},
        registry=registry,
        expected_input_dim=2381,
        validation_metrics={"accuracy": 0.93, "f1": 0.92},
    )
    assert e2.status == "ACTIVE"
    assert registry.get("fl-v1").status == "ARCHIVED"
    assert registry.get_active().version == "fl-v2"


def test_candidate_validation_failure_previous_remains(tmp_path):
    """Candidate → validation failure → previous model remains active."""
    ckpt_good = _dummy_checkpoint(tmp_path, "good.pt")
    ckpt_bad = tmp_path / "bad.pt"
    # Create a checkpoint that will fail validation due to missing metrics? We'll use empty metrics
    import torch
    from src.federated.models.mlp import MLPConfig, build_mlp
    cfg = MLPConfig(input_dim=2381, hidden_layers=(256, 128))
    m = build_mlp(cfg)
    torch.save(m.state_dict(), ckpt_bad)

    schema = _make_schema()
    registry = ModelRegistry(tmp_path / "registry")

    # Good first
    fl_checkpoint_to_registry(
        checkpoint_path=ckpt_good,
        model_cfg=cfg,
        scaler_path=Path("nonexistent"),
        metrics={"accuracy": 0.90},
        version="good-v1",
        algorithm="fedavg",
        training_round=5,
        feature_schema=schema,
        preprocessing_version=schema.preprocessing_version,
        configuration={"model": {"input_dim": 2381}},
        registry=registry,
        expected_input_dim=2381,
        validation_metrics={"accuracy": 0.90, "f1": 0.89},
    )
    assert registry.get_active().version == "good-v1"

    # Bad candidate: no validation metrics → validation fails → REJECTED, previous remains
    bad_schema = _make_schema()
    bad_meta_empty = ModelMetadata(
        version="bad-v2",
        algorithm="fedavg",
        training_round=6,
        feature_schema=bad_schema,
        metrics={},  # empty → evaluation should fail
        created_at="2026-01-01T00:00:00",
        num_parameters=642817,
        input_dim=2381,
    )
    entry = registry.register(metadata=bad_meta_empty, artifact_source=ckpt_bad, expected_input_dim=2381, configuration={})
    # Try to validate with empty metrics - should fail (evaluation succeeded check)
    valid, issues = registry.validate("bad-v2")
    assert not valid
    # Try fl_checkpoint_to_registry with empty validation_metrics
    # It will reject and keep previous active
    result = fl_checkpoint_to_registry(
        checkpoint_path=ckpt_bad,
        model_cfg=cfg,
        scaler_path=Path("nonexistent"),
        metrics={},
        version="bad-v2-2",
        algorithm="fedavg",
        training_round=6,
        feature_schema=schema,
        preprocessing_version=schema.preprocessing_version,
        configuration={},
        registry=registry,
        expected_input_dim=2381,
        validation_metrics={},  # empty → evaluation should fail
    )
    assert result.status == "REJECTED"
    assert registry.get_active().version == "good-v1"


def test_incompatible_feature_schema_reject(tmp_path):
    """Candidate → incompatible feature schema → reject."""
    ckpt = _dummy_checkpoint(tmp_path, "ckpt.pt")
    good_schema = _make_schema(preprocessing_version="ember_v17_std")
    bad_schema = FeatureSchema(
        feature_names=tuple(f"bad_{i}" for i in range(2381)),
        feature_types=("float32",) * 2381,
        preprocessing_version="ember_v99_std",
        created_at="2026-01-01T00:00:00",
        model_version="bad",
    )
    registry = ModelRegistry(tmp_path / "registry")
    # First, establish a good active
    from src.federated.models.mlp import MLPConfig
    cfg = MLPConfig(input_dim=2381, hidden_layers=(256, 128))
    fl_checkpoint_to_registry(
        checkpoint_path=ckpt,
        model_cfg=cfg,
        scaler_path=Path("nonexistent"),
        metrics={"accuracy": 0.90},
        version="good-v1",
        algorithm="fedavg",
        training_round=5,
        feature_schema=good_schema,
        preprocessing_version=good_schema.preprocessing_version,
        configuration={},
        registry=registry,
        expected_input_dim=2381,
        validation_metrics={"accuracy": 0.90, "f1": 0.89},
    )
    assert registry.get_active().version == "good-v1"

    # Try to register bad schema but validate against good expected schema
    bad_meta = _make_metadata("bad-schema-v2", bad_schema)
    entry = registry.register(metadata=bad_meta, artifact_source=ckpt, expected_input_dim=2381)
    # Validate against good schema should fail
    valid, issues = registry.validate("bad-schema-v2", expected_schema=good_schema, expected_preprocessing=good_schema.preprocessing_version)
    assert not valid
    assert any("schema" in i.lower() or "preprocessing" in i.lower() for i in issues)
    # Mark as rejected due to incompatible schema (activation would be blocked by validation)
    registry.reject("bad-schema-v2", reason="incompatible feature schema")
    assert registry.get("bad-schema-v2").status == "REJECTED"
    # Previous remains active
    assert registry.get_active().version == "good-v1"


def test_corrupted_model_reject(tmp_path):
    """Candidate → corrupted model → reject."""
    ckpt_good = _dummy_checkpoint(tmp_path, "good.pt")
    ckpt_corrupt = tmp_path / "corrupt.pt"
    ckpt_corrupt.write_bytes(b"not a torch file - corrupted")
    schema = _make_schema()
    registry = ModelRegistry(tmp_path / "registry")
    from src.federated.models.mlp import MLPConfig
    cfg = MLPConfig(input_dim=2381, hidden_layers=(256, 128))

    fl_checkpoint_to_registry(
        checkpoint_path=ckpt_good,
        model_cfg=cfg,
        scaler_path=Path("nonexistent"),
        metrics={"accuracy": 0.90},
        version="good-v1",
        algorithm="fedavg",
        training_round=5,
        feature_schema=schema,
        preprocessing_version=schema.preprocessing_version,
        configuration={},
        registry=registry,
        expected_input_dim=2381,
        validation_metrics={"accuracy": 0.90, "f1": 0.89},
    )

    # Corrupted candidate
    result = fl_checkpoint_to_registry(
        checkpoint_path=ckpt_corrupt,
        model_cfg=cfg,
        scaler_path=Path("nonexistent"),
        metrics={"accuracy": 0.90},
        version="corrupt-v2",
        algorithm="fedavg",
        training_round=6,
        feature_schema=schema,
        preprocessing_version=schema.preprocessing_version,
        configuration={},
        registry=registry,
        expected_input_dim=2381,
        validation_metrics={"accuracy": 0.90, "f1": 0.89},
    )
    assert result.status == "REJECTED"
    assert registry.get_active().version == "good-v1"


def test_automatic_endpoint_update_and_rollback(tmp_path):
    """Automatic endpoint update: discover ACTIVE, verify, deploy; rollback on rejection."""
    ckpt1 = _dummy_checkpoint(tmp_path, "v1.pt")
    ckpt2 = _dummy_checkpoint(tmp_path, "v2.pt")
    schema = _make_schema()
    registry = ModelRegistry(tmp_path / "registry")
    from src.federated.models.mlp import MLPConfig
    cfg = MLPConfig(input_dim=2381, hidden_layers=(256, 128))

    # v1 active
    fl_checkpoint_to_registry(
        checkpoint_path=ckpt1,
        model_cfg=cfg,
        scaler_path=Path("nonexistent"),
        metrics={"accuracy": 0.90},
        version="v1",
        algorithm="fedavg",
        training_round=5,
        feature_schema=schema,
        preprocessing_version=schema.preprocessing_version,
        configuration={},
        registry=registry,
        expected_input_dim=2381,
        validation_metrics={"accuracy": 0.90, "f1": 0.89},
    )
    endpoint_dir = tmp_path / "endpoint"
    # First discovery should deploy v1
    entry = endpoint_discover_and_update(registry, endpoint_dir, current_version=None)
    assert entry is not None and entry.version == "v1"
    assert (endpoint_dir / "model.pt").exists()
    assert (endpoint_dir / "active_version.txt").read_text() == "v1"

    # v2 becomes active
    fl_checkpoint_to_registry(
        checkpoint_path=ckpt2,
        model_cfg=cfg,
        scaler_path=Path("nonexistent"),
        metrics={"accuracy": 0.93},
        version="v2",
        algorithm="fedavg",
        training_round=10,
        feature_schema=schema,
        preprocessing_version=schema.preprocessing_version,
        configuration={},
        registry=registry,
        expected_input_dim=2381,
        validation_metrics={"accuracy": 0.93, "f1": 0.92},
    )
    # Updater should discover v2
    updater = EndpointModelUpdater(registry, endpoint_dir)
    new_ver = updater.check_and_update()
    assert new_ver == "v2"
    assert updater.current_version == "v2"
    assert (endpoint_dir / "active_version.txt").read_text() == "v2"

    # Simulate rejection of v2 (later found invalid) → should auto-rollback to v1 via reject()
    registry.reject("v2", reason="later found invalid")
    # reject() already triggers rollback to previous trusted (v1)
    assert registry.get_active().version == "v1"
    assert registry.get("v1").status == "ACTIVE"
    # Endpoint should now discover rollback (v1 again)
    # Reset updater's current to simulate stale endpoint still on v2
    updater._current_version = "v2"
    entry2 = endpoint_discover_and_update(registry, endpoint_dir, current_version="v2")
    assert entry2.version == "v1"
    assert (endpoint_dir / "active_version.txt").read_text() == "v1"


def test_endpoint_continues_when_server_unavailable(tmp_path):
    """If network/server is unavailable, endpoint continues using current active model."""
    ckpt = _dummy_checkpoint(tmp_path, "v1.pt")
    schema = _make_schema()
    registry = ModelRegistry(tmp_path / "registry")
    from src.federated.models.mlp import MLPConfig
    cfg = MLPConfig(input_dim=2381, hidden_layers=(256, 128))
    fl_checkpoint_to_registry(
        checkpoint_path=ckpt,
        model_cfg=cfg,
        scaler_path=Path("nonexistent"),
        metrics={"accuracy": 0.90},
        version="v1",
        algorithm="fedavg",
        training_round=5,
        feature_schema=schema,
        preprocessing_version=schema.preprocessing_version,
        configuration={},
        registry=registry,
        expected_input_dim=2381,
        validation_metrics={"accuracy": 0.90, "f1": 0.89},
    )
    endpoint_dir = tmp_path / "endpoint"
    endpoint_discover_and_update(registry, endpoint_dir, current_version=None)
    assert (endpoint_dir / "model.pt").exists()

    # Simulate server unavailable: use a registry path that doesn't exist / unreadable
    # Updater should gracefully return None and keep current
    updater = EndpointModelUpdater(registry, endpoint_dir)
    updater._current_version = "v1"
    # Corrupt the registry index to simulate failure? Instead, test with a missing registry
    missing_registry = ModelRegistry(tmp_path / "missing_registry_no_active")
    # This registry has no active model, updater should return None and keep v1
    updater2 = EndpointModelUpdater(missing_registry, endpoint_dir, poll_interval_s=0.1)
    updater2._current_version = "v1"
    result = updater2.check_and_update()
    assert result is None
    assert (endpoint_dir / "active_version.txt").read_text() == "v1"
    assert updater2.current_version == "v1"
