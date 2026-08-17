"""Tests for the model registry: registration, validation, approval lifecycle."""

import pytest

from src.federated.model_registry import ModelRegistry
from src.interfaces import FeatureSchema, ModelMetadata


def _schema() -> FeatureSchema:
    return FeatureSchema(
        feature_names=tuple(f"f{i}" for i in range(2381)),
        feature_types=("float32",) * 2381,
        preprocessing_version="prep-v1",
        created_at="2026-01-01T00:00:00",
        model_version="v1",
    )


def _metadata(version: str, algorithm: str = "fedavg", input_dim: int = 2381) -> ModelMetadata:
    return ModelMetadata(
        version=version,
        algorithm=algorithm,
        training_round=5,
        feature_schema=_schema(),
        metrics={"accuracy": 0.9},
        created_at="2026-01-01T00:00:00",
        num_parameters=1000,
        input_dim=input_dim,
    )


def test_register_approve_activate(tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"weights")
    registry = ModelRegistry(tmp_path / "registry")

    entry = registry.register(_metadata("v1"), artifact)
    assert entry.status == "pending"
    assert registry.get_active() is None  # not auto-activated

    registry.approve("v1")
    active = registry.get_active()
    assert active is not None
    assert active.version == "v1"
    assert active.status == "approved"
    assert (tmp_path / "registry" / "active.txt").exists()


def test_duplicate_version_rejected(tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"w")
    registry = ModelRegistry(tmp_path / "registry")
    registry.register(_metadata("v1"), artifact)
    with pytest.raises(ValueError):
        registry.register(_metadata("v1"), artifact)


def test_approve_unknown_version_raises(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    with pytest.raises(KeyError):
        registry.approve("nope")


def test_cannot_approve_missing_artifact(tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"w")
    registry = ModelRegistry(tmp_path / "registry")
    registry.register(_metadata("v1"), artifact)
    # Delete the copied artifact to simulate corruption
    import shutil

    entry = registry.get("v1")
    shutil.rmtree(str(entry.artifact_path).replace("model.pt", ""))
    with pytest.raises(ValueError):
        registry.approve("v1")
    # and it can be rejected instead
    registry.reject("v1", reason="artifact missing")
    assert registry.get("v1").status == "rejected"


def test_approving_new_supersedes_old(tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"w")
    registry = ModelRegistry(tmp_path / "registry")
    registry.register(_metadata("v1"), artifact)
    registry.approve("v1")
    registry.register(_metadata("v2"), artifact)
    registry.approve("v2")
    assert registry.get("v1").status == "superseded"
    assert registry.get_active().version == "v2"


def test_persistence_across_instances(tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"w")
    registry = ModelRegistry(tmp_path / "registry")
    registry.register(_metadata("v1"), artifact)
    registry.approve("v1")

    reloaded = ModelRegistry(tmp_path / "registry")
    assert reloaded.get_active().version == "v1"
    assert reloaded.status_summary() == {"approved": 1}


def test_input_dim_mismatch_blocked_at_approval(tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"w")
    registry = ModelRegistry(tmp_path / "registry")
    entry = registry.register(_metadata("v1", input_dim=100), artifact, expected_input_dim=2381)
    assert "input_dim mismatch" in entry.validation_notes
    # an incompatible model must NOT be activated
    with pytest.raises(ValueError):
        registry.approve("v1")
    assert registry.get_active() is None
