"""Verify all package modules import without errors."""

import importlib

MODULES = [
    # core
    "fedshield.config",
    "fedshield.logging_setup",
    # interfaces
    "src.interfaces",
    # endpoint engine
    "src.endpoint.monitor",
    "src.endpoint.file_analysis",
    "src.endpoint.feature_extraction",
    "src.endpoint.inference",
    "src.endpoint.risk",
    "src.endpoint.quarantine",
    "src.endpoint.notifications",
    "src.endpoint.history",
    # federated engine
    "src.federated.model_registry",
    "src.federated.data",
    "src.federated.models",
    "src.federated.training",
    "src.federated.algorithms",
    "src.federated.evaluation",
    "src.federated.experiments",
    "src.federated.privacy",
    "src.federated.security",
    # utils
    "src.utils.reproducibility",
]


def test_all_modules_import():
    for module in MODULES:
        importlib.import_module(module)


def test_interfaces_roundtrip():
    from src.interfaces import DetectionRecord, FeatureSchema

    schema = FeatureSchema(
        feature_names=("f1", "f2"),
        feature_types=("float32", "float32"),
        preprocessing_version="prep-v1",
        created_at="2026-01-01T00:00:00",
        model_version="v1",
    )
    record = DetectionRecord(
        detection_id="abc123",
        timestamp="2026-01-01T00:00:00",
        filename="a.exe",
        filepath="C:/x/a.exe",
        sha256="ab" * 32,
        file_type="pe_exe",
        model_version="v1",
        malware_probability=0.9,
        benign_probability=0.1,
        risk_score=90,
        risk_level="HIGH",
        verdict="HIGH",
        action="QUARANTINE",
        model_algorithm="fedavg",
        analysis_duration_ms=12.5,
    )
    restored = DetectionRecord.from_dict(record.to_dict())
    assert restored == record
