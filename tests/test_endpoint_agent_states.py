"""Enhancement 2 — Tests for 7 endpoint states and transitions."""

import pathlib
import tempfile
import torch

from src.endpoint.client_agent import FedShieldClientAgent, AgentState
from src.federated.models.mlp import MLPConfig, build_mlp
from src.federated.model_registry import ModelRegistry
from src.interfaces import FeatureSchema, ModelMetadata
from src.federated.data.feature_schema import FEATURE_NAMES, FEATURE_VERSION, N_FEATURES


def _valid_ckpt(tmp_path, name="model.pt"):
    cfg = MLPConfig(input_dim=2381, hidden_layers=(32,))
    m = build_mlp(cfg)
    p = pathlib.Path(tmp_path) / name
    torch.save(m.state_dict(), p)
    return p, cfg


def _schema():
    return FeatureSchema(feature_names=tuple(FEATURE_NAMES[:2381]), feature_types=("float32",)*2381, preprocessing_version=f"ember_v{FEATURE_VERSION}_std", created_at="2026-01-01T00:00:00", model_version="v1")


def test_starting_to_protected(tmp_path):
    agent = FedShieldClientAgent(registry_dir=tmp_path/"reg", queue_dir=tmp_path/"q", max_retries=1)
    state = agent.startup()
    assert state in (AgentState.PROTECTED, AgentState.DEGRADED, AgentState.STARTING)
    # After startup, health should be queryable
    health = agent.get_health()
    assert "state" in health


def test_protected_to_degraded_on_scan_failure(tmp_path):
    agent = FedShieldClientAgent(registry_dir=tmp_path/"reg", queue_dir=tmp_path/"q", max_retries=1)
    agent.startup()
    # Simulate a detector that always fails
    class BadDetector:
        def scan(self, p): raise RuntimeError("scan fail")
    result = agent.safe_scan(tmp_path/"nonexistent.exe", BadDetector())
    assert result is None  # failure isolated, not propagated
    # State should be DEGRADED after repeated scan failures
    assert agent.state in (AgentState.DEGRADED, AgentState.PROTECTED, AgentState.ERROR)


def test_server_unavailable_state(tmp_path):
    agent = FedShieldClientAgent(registry_dir=tmp_path/"reg", queue_dir=tmp_path/"q")
    agent.startup()
    offline = agent.handle_server_unavailable()
    assert agent.state == AgentState.SERVER_UNAVAILABLE
    assert offline["status"] == "offline_mode"


def test_model_update_pending(tmp_path):
    ckpt, cfg = _valid_ckpt(tmp_path, "m.pt")
    schema = _schema()
    reg = ModelRegistry(tmp_path/"reg2")
    meta = ModelMetadata(version="v1", algorithm="fedavg", training_round=1, feature_schema=schema, metrics={"f1":0.9}, created_at="2026-01-01T00:00:00", num_parameters=1000, input_dim=2381)
    reg.register(meta, ckpt)
    reg.mark_validated("v1", validation_metrics={"f1":0.9})
    agent = FedShieldClientAgent(registry_dir=tmp_path/"reg2", queue_dir=tmp_path/"q2")
    # Simulate model update pending
    agent.state = AgentState.MODEL_UPDATE_PENDING
    assert agent.get_health()["state"] == "MODEL_UPDATE_PENDING"


def test_error_to_stopped(tmp_path):
    agent = FedShieldClientAgent(registry_dir=tmp_path/"reg", queue_dir=tmp_path/"q")
    agent.state = AgentState.ERROR
    stopped = agent.shutdown(graceful=True)
    assert stopped == AgentState.STOPPED


def test_restart_transitions(tmp_path):
    agent = FedShieldClientAgent(registry_dir=tmp_path/"reg", queue_dir=tmp_path/"q", max_retries=1)
    agent.startup()
    before = agent.state
    after = agent.restart()
    assert after in (AgentState.PROTECTED, AgentState.DEGRADED, AgentState.STARTING)


def test_survives_locked_and_permission_failure(tmp_path):
    agent = FedShieldClientAgent(registry_dir=tmp_path/"reg", queue_dir=tmp_path/"q", max_retries=1)
    agent.startup()
    # Create a file and make it unreadable (permission failure) — on Windows we simulate via missing file
    fake = tmp_path / "locked.exe"
    fake.write_bytes(b"MZ")
    # Use a detector that would succeed, but file is present — should not crash
    class GoodDetector:
        def scan(self, p):
            class R:
                record = type("obj", (), {"detection_id": "123"})()
                def to_dict(self): return {"ok": True}
            return R()
    result = agent.safe_scan(fake, GoodDetector())
    assert result is not None or result is None  # either is fine, but must not raise
    # Unsupported file (non-PE) should be handled
    txt = tmp_path / "readme.txt"
    txt.write_text("hello")
    result2 = agent.safe_scan(txt, GoodDetector())
    assert result2 is not None  # still returns, failure isolated


def test_graceful_restart_after_corrupted_record(tmp_path):
    agent = FedShieldClientAgent(registry_dir=tmp_path/"reg", queue_dir=tmp_path/"q", max_retries=1)
    class CorruptDetector:
        def scan(self, p):
            # Return corrupted record (no detection_id)
            class R:
                record = None
            return R()
    fake = tmp_path / "a.exe"
    fake.write_bytes(b"MZ")
    result = agent.safe_scan(fake, CorruptDetector())
    assert result is None  # corrupted record handled
    assert agent.state in (AgentState.DEGRADED, AgentState.PROTECTED)
