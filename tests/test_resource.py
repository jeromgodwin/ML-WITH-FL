"""Phase 12: resource-aware federated training tests.

Covers the resource monitor (graceful unsupported metrics), the policy
(config-driven decisions with no hardcoded thresholds), the training
controller state machine (start/pause/resume/cancel), FL-client gating, and
the architectural rule that real-time detection NEVER blocks on training:
the detection pipeline stays fully operational while training is paused,
and training cannot be blocked by detection.
"""

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedshield.config import ResourceConfig  # noqa: E402
from src.endpoint.detector import AutoDetector  # noqa: E402
from src.endpoint.feature_extraction import FeatureExtractor, FeatureVector  # noqa: E402
from src.endpoint.resource import (  # noqa: E402
    STATE_CANCELLED,
    STATE_FINISHED,
    STATE_IDLE,
    STATE_PAUSED,
    STATE_STARTED,
    ResourceMonitor,
    ResourcePolicy,
    TrainingController,
    create_controller_from_config,
)
from src.endpoint.resource.policy import PolicyDecision  # noqa: E402
from src.endpoint.risk import RiskEngine  # noqa: E402
from src.federated.data.feature_schema import FEATURE_NAMES, N_FEATURES  # noqa: E402
from src.federated.fl.client import FedAvgClient  # noqa: E402
from src.federated.models.mlp import MLPConfig  # noqa: E402

RESOURCE_CONFIG = ResourceConfig(
    enabled=True,
    max_cpu_percent=50.0,
    min_battery_percent=20.0,
    require_ac_power=False,
    idle_only=False,
    idle_min_seconds=60.0,
    max_training_duration_sec=None,
    min_free_memory_mb=512.0,
    check_interval_sec=0.01,
)


def permit_snapshot(cpu=10.0, ram_free_mb=4096.0, battery=90.0, ac=True,
                    idle=120.0):
    return {
        "cpu_percent": cpu, "cpu_available": True,
        "ram_used_mb": 1024.0, "ram_total_mb": 8192.0, "ram_free_mb": ram_free_mb,
        "ram_percent": 12.5, "ram_available": True,
        "battery_percent": battery, "ac_powered": ac, "battery_available": True,
        "idle_seconds": idle, "user_active": idle < 60, "activity_available": True,
    }


def make_monitor(**providers):
    return ResourceMonitor(providers=providers)


def make_controller(config=RESOURCE_CONFIG, monitor=None, interval=None):
    return TrainingController(
        ResourcePolicy(config),
        monitor or make_monitor(cpu=lambda: 10.0),
        check_interval_sec=interval or config.check_interval_sec,
    )


# ---------------------------------------------------------------------------
# monitor: graceful unsupported metrics
# ---------------------------------------------------------------------------

def test_monitor_all_metrics_present_with_providers():
    mon = make_monitor(cpu=lambda: 42.0,
                       ram=lambda: (1 << 20, 8 << 20, 7 << 20),
                       battery=lambda: (87.0, True),
                       idle=lambda: 123.0)
    s = mon.snapshot()
    assert s["cpu_percent"] == 42.0 and s["cpu_available"]
    assert s["ram_used_mb"] == 1.0 and s["ram_total_mb"] == 8.0
    assert s["ram_free_mb"] == 7.0 and s["ram_available"]
    assert s["battery_percent"] == 87.0 and s["ac_powered"] is True
    assert s["battery_available"]
    assert s["idle_seconds"] == 123.0 and s["activity_available"]


def test_monitor_unsupported_metrics_degrade_gracefully():
    mon = make_monitor(cpu=lambda: None, ram=lambda: None,
                       battery=lambda: None, idle=lambda: None)
    s = mon.snapshot()
    assert s["cpu_percent"] is None and not s["cpu_available"]
    assert s["ram_used_mb"] is None and not s["ram_available"]
    assert s["battery_percent"] is None and not s["battery_available"]
    assert s["idle_seconds"] is None and not s["activity_available"]


def test_monitor_failing_provider_degrades_gracefully():
    def boom():
        raise OSError("no such device")
    mon = make_monitor(cpu=boom)
    s = mon.snapshot()
    assert s["cpu_percent"] is None and not s["cpu_available"]


def test_monitor_missing_battery_is_unsupported():
    mon = make_monitor(battery=lambda: None)
    s = mon.snapshot()
    assert not s["battery_available"]
    assert s["battery_percent"] is None and s["ac_powered"] is None


# ---------------------------------------------------------------------------
# policy: config-driven decisions
# ---------------------------------------------------------------------------

def test_policy_permit_when_within_bounds():
    assert ResourcePolicy(RESOURCE_CONFIG).decide(permit_snapshot(), 1.0) \
        == PolicyDecision("permit", "", ())


def test_policy_pause_on_high_cpu():
    d = ResourcePolicy(RESOURCE_CONFIG).decide(permit_snapshot(cpu=90.0), 1.0)
    assert d.action == "pause" and d.reason == "high_cpu"


def test_policy_pause_on_low_battery():
    d = ResourcePolicy(RESOURCE_CONFIG).decide(permit_snapshot(battery=10.0), 1.0)
    assert d.action == "pause" and d.reason == "low_battery"


def test_policy_pause_on_battery_when_ac_required():
    cfg = RESOURCE_CONFIG.__class__(**{**RESOURCE_CONFIG.__dict__,
                                       "require_ac_power": True})
    d = ResourcePolicy(cfg).decide(permit_snapshot(ac=False), 1.0)
    assert d.action == "pause" and d.reason == "on_battery"


def test_policy_pause_on_active_user_when_idle_only():
    cfg = RESOURCE_CONFIG.__class__(**{**RESOURCE_CONFIG.__dict__,
                                       "idle_only": True})
    d = ResourcePolicy(cfg).decide(permit_snapshot(idle=5.0), 1.0)
    assert d.action == "pause" and d.reason == "user_active"
    assert ResourcePolicy(cfg).decide(permit_snapshot(idle=120.0), 1.0).action == "permit"


def test_policy_pause_on_low_memory():
    d = ResourcePolicy(RESOURCE_CONFIG).decide(permit_snapshot(ram_free_mb=64.0), 1.0)
    assert d.action == "pause" and d.reason == "low_memory"


def test_policy_cancel_on_max_duration():
    d = ResourcePolicy(RESOURCE_CONFIG).decide(permit_snapshot(), 999.0)
    assert d.action == "permit"  # no duration limit configured -> never cancels
    cfg = RESOURCE_CONFIG.__class__(**{**RESOURCE_CONFIG.__dict__,
                                       "max_training_duration_sec": 10.0})
    d = ResourcePolicy(cfg).decide(permit_snapshot(), 11.0)
    assert d.action == "cancel" and d.reason == "max_duration"


def test_policy_disabled_always_permits():
    cfg = RESOURCE_CONFIG.__class__(**{**RESOURCE_CONFIG.__dict__, "enabled": False})
    for snap in (permit_snapshot(cpu=99.0, battery=1.0), {}):
        assert ResourcePolicy(cfg).decide(snap, 1.0).action == "permit"


def test_policy_unset_constraints_never_applied():
    cfg = ResourceConfig(enabled=True, max_cpu_percent=None,
                         min_battery_percent=None, min_free_memory_mb=None)
    d = ResourcePolicy(cfg).decide(permit_snapshot(cpu=99.0, battery=1.0,
                                                   ram_free_mb=1.0), 1.0)
    assert d.action == "permit"


def test_policy_unavailable_metric_skips_that_constraint():
    snap = permit_snapshot()
    snap["cpu_percent"], snap["cpu_available"] = None, False
    d = ResourcePolicy(RESOURCE_CONFIG).decide(snap, 1.0)
    assert d.action == "permit"
    assert "cpu" in d.unavailable


def test_policy_thresholds_come_from_config_not_code():
    cfg = RESOURCE_CONFIG.__class__(**{**RESOURCE_CONFIG.__dict__,
                                       "max_cpu_percent": 95.0})
    d = ResourcePolicy(cfg).decide(permit_snapshot(cpu=90.0), 1.0)
    assert d.action == "permit"  # 90 < 95 -> the SAME snapshot that paused
                                 # under the default 50 must now pass


# ---------------------------------------------------------------------------
# controller: state machine and gate
# ---------------------------------------------------------------------------

def test_controller_state_machine():
    ctl = make_controller()
    assert ctl.status()["state"] == STATE_IDLE
    assert ctl.request_start()
    assert ctl.status()["state"] == STATE_STARTED
    assert not ctl.request_start()  # already started
    assert ctl.pause("high_cpu")
    assert ctl.status()["state"] == STATE_PAUSED
    assert not ctl.pause("high_cpu")  # idempotent
    assert ctl.resume()
    assert ctl.status()["state"] == STATE_STARTED
    assert not ctl.resume()  # no-op when not paused
    assert ctl.cancel("manual")
    assert ctl.status()["state"] == STATE_CANCELLED
    assert not ctl.pause("manual")
    assert not ctl.resume()
    ctl2 = make_controller()
    ctl2.request_start()
    ctl2.finish()
    assert ctl2.status()["state"] == STATE_FINISHED


def test_controller_pause_records_reason_and_count():
    ctl = make_controller()
    ctl.request_start()
    ctl.pause("high_cpu")
    ctl.resume()
    ctl.pause("low_battery")
    st = ctl.status()
    assert st["pauses"] == 2
    assert st["last_pause_reason"] == "low_battery"
    assert st["paused_total_sec"] >= 0.0
    json.dumps(st)  # JSON-serializable for the dashboard


def test_controller_wait_blocks_while_paused_then_resumes():
    ctl = make_controller()
    ctl.request_start()
    ctl.pause("manual")
    result = {}
    t = threading.Thread(target=lambda: result.setdefault(
        "allowed", ctl.wait_until_allowed(timeout=5.0)))
    t.start()
    time.sleep(0.05)
    assert not result  # still waiting while paused
    ctl.resume()
    t.join(timeout=5.0)
    assert result.get("allowed") is True


def test_controller_wait_returns_false_on_cancel():
    ctl = make_controller()
    ctl.request_start()
    ctl.pause("manual")
    result = {}
    t = threading.Thread(target=lambda: result.setdefault(
        "allowed", ctl.wait_until_allowed(timeout=5.0)))
    t.start()
    time.sleep(0.05)
    ctl.cancel("user")
    t.join(timeout=5.0)
    assert result.get("allowed") is False


def test_controller_wait_waits_out_high_cpu_then_permits():
    state = {"cpu": 90.0}
    mon = make_monitor(cpu=lambda: state["cpu"])
    ctl = make_controller(monitor=mon)
    ctl.request_start()
    result = {}
    t = threading.Thread(target=lambda: result.setdefault(
        "allowed", ctl.wait_until_allowed(timeout=5.0)))
    t.start()
    time.sleep(0.05)
    assert not result  # high CPU -> deferred
    state["cpu"] = 10.0  # contention clears
    t.join(timeout=5.0)
    assert result.get("allowed") is True
    assert ctl.status()["pauses"] >= 1
    assert ctl.status()["last_pause_reason"] == "high_cpu"


def test_controller_max_duration_cancels_training():
    cfg = RESOURCE_CONFIG.__class__(**{**RESOURCE_CONFIG.__dict__,
                                       "max_training_duration_sec": 0.0})
    ctl = make_controller(config=cfg)
    ctl.request_start()
    assert ctl.wait_until_allowed(timeout=2.0) is False
    assert ctl.status()["state"] == STATE_CANCELLED


def test_create_controller_from_config():
    ctl = create_controller_from_config(RESOURCE_CONFIG)
    ctl.request_start()
    assert ctl.wait_until_allowed(timeout=2.0) is True


# ---------------------------------------------------------------------------
# architectural rule: detection never blocked by training
# ---------------------------------------------------------------------------

class FakeBundle:
    def __init__(self, p=0.01):
        self.p = p
        self.manifest = {"version": "fake-v1", "algorithm": "fake"}

    def predict_proba(self, X):
        return np.full(X.shape[0], self.p, dtype=np.float32)


class FakeExtractor:
    def extract(self, path):
        return FeatureVector(
            features=np.zeros(N_FEATURES, dtype=np.float32),
            feature_names=FEATURE_NAMES, schema_version="ember_v2_std",
            model_version="fake-v1", extraction_success=True,
            missing_features=[], extra_features=[],
            sha256="0" * 64, file_path=str(path), file_size=1,
            extracted_at="2026-01-01T00:00:00+00:00",
        )


def test_detection_works_while_training_paused(tmp_path):
    """Real-time detection must stay operational while FL training is paused."""
    detector = AutoDetector(FakeBundle(p=0.01), FakeExtractor(), RiskEngine())
    target = tmp_path / "sample.exe"
    target.write_bytes(b"MZ" + b"\x00" * 128)

    ctl = make_controller()
    ctl.request_start()
    ctl.pause("high_cpu")

    result = detector.scan(target)
    assert result.record.verdict == "LOW"
    assert result.record.action == "ALLOW"
    assert result.record.analysis_duration_ms > 0
    assert ctl.status()["state"] == STATE_PAUSED  # training still paused,
                                                  # detection unaffected


def test_detection_works_while_training_waits_for_resources(tmp_path):
    """A training thread blocked on the resource gate must not stall scanning."""
    state = {"cpu": 95.0}
    mon = make_monitor(cpu=lambda: state["cpu"])
    ctl = make_controller(monitor=mon)
    ctl.request_start()
    waiter = threading.Thread(target=ctl.wait_until_allowed,
                              kwargs={"timeout": 5.0}, daemon=True)
    waiter.start()
    time.sleep(0.05)

    detector = AutoDetector(FakeBundle(p=0.9), FakeExtractor(), RiskEngine())
    target = tmp_path / "risk.exe"
    target.write_bytes(b"MZ" + b"\x00" * 128)
    result = detector.scan(target)
    assert result.record.verdict == "HIGH"
    assert result.record.action == "QUARANTINE"

    state["cpu"] = 5.0
    waiter.join(timeout=5.0)
    assert not waiter.is_alive()


def test_pause_does_not_touch_detection_pipeline_objects(tmp_path):
    """The controller has no handles on detection components (one-way rule)."""
    ctl = make_controller()
    ctl.request_start()
    ctl.pause("manual")
    # the controller's only dependencies are policy + monitor
    assert isinstance(ctl.policy, ResourcePolicy)
    assert isinstance(ctl.monitor, ResourceMonitor)


# ---------------------------------------------------------------------------
# FL client gating
# ---------------------------------------------------------------------------

def _tiny_client(controller=None, n=256, d=32):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    cfg = MLPConfig(input_dim=d, hidden_layers=(16, 8), dropout=0.0)
    return FedAvgClient(cfg, X[:200], y[:200], X[200:], y[200:],
                        seed=42, controller=controller)


def test_client_unrestricted_without_controller():
    client = _tiny_client()
    params, n, metrics = client.fit(
        client.get_parameters({}), {"local_epochs": 2, "lr": 1e-3})
    assert "resource_gated" not in metrics


def test_client_fit_deferred_by_high_cpu_then_resumes():
    state = {"cpu": 95.0}
    mon = make_monitor(cpu=lambda: state["cpu"])
    ctl = make_controller(monitor=mon)
    ctl.request_start()
    client = _tiny_client(controller=ctl)

    def clear_cpu():
        time.sleep(0.05)
        state["cpu"] = 5.0
    threading.Thread(target=clear_cpu, daemon=True).start()

    t0 = time.perf_counter()
    params, n, metrics = client.fit(
        client.get_parameters({}), {"local_epochs": 2, "lr": 1e-3})
    assert metrics["resource_gated"] is True
    assert metrics["aborted"] is False
    assert metrics["epochs_completed"] == 2
    assert metrics["gate_wait_ms"] > 0
    assert (time.perf_counter() - t0) >= 0.05  # actually waited


def test_client_fit_aborts_when_training_cancelled():
    ctl = make_controller()
    ctl.request_start()
    client = _tiny_client(controller=ctl)

    def cancel_soon():
        time.sleep(0.02)
        ctl.cancel("user")
    threading.Thread(target=cancel_soon, daemon=True).start()

    params, n, metrics = client.fit(
        client.get_parameters({}), {"local_epochs": 10, "lr": 1e-3})
    assert metrics["resource_gated"] is True
    assert metrics["aborted"] is True
    assert metrics["epochs_completed"] < 10
    assert len(params) > 0  # partial result is still returned to the server


def test_client_max_duration_aborts_fit():
    cfg = RESOURCE_CONFIG.__class__(**{**RESOURCE_CONFIG.__dict__,
                                       "max_training_duration_sec": 0.0})
    ctl = make_controller(config=cfg)
    ctl.request_start()
    client = _tiny_client(controller=ctl)
    params, n, metrics = client.fit(
        client.get_parameters({}), {"local_epochs": 3, "lr": 1e-3})
    assert metrics["aborted"] is True
    assert metrics["epochs_completed"] == 0
