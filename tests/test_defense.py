"""Phase 14 tests: poisoning defenses and simulated attacks (synthetic only)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from fedshield.config import AttackConfig, DefenseConfig, ExperimentConfig
from src.federated.defense.anomaly import (
    AnomalyClass,
    UpdateAnomalyDetector,
)
from src.federated.defense.attack import (
    AttackSpec,
    apply_attack,
    flip_labels,
    is_malicious_cid,
)
from src.federated.defense.clipping import UpdateClipper
from src.federated.defense.robust import coordinate_wise_median, trimmed_mean
from src.federated.defense.validation import ValidationDecision, ValidationGate
from src.federated.fl.strategy import DefendedTracked


# ----------------------------------------------------------------------
# attack simulation
# ----------------------------------------------------------------------

def test_attack_spec_defaults():
    spec = AttackSpec()
    assert spec.is_malicious is False
    assert spec.attack_type == "none"
    assert spec.to_dict()["enabled"] is False


def test_is_malicious_cid():
    spec = AttackSpec(enabled=True, attack_type="scaled_update", n_malicious=2)
    assert is_malicious_cid(0, spec)
    assert is_malicious_cid(1, spec)
    assert not is_malicious_cid(2, spec)
    assert not is_malicious_cid(0, AttackSpec())


def test_flip_labels_flips_exact_fraction():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=1000)
    flipped = flip_labels(y, 0.5, seed=7)
    n_changed = int((flipped != y).sum())
    assert n_changed == 500
    # deterministic
    assert np.array_equal(flipped, flip_labels(y, 0.5, seed=7))


def test_apply_attack_scaled_update():
    spec = AttackSpec(enabled=True, attack_type="scaled_update",
                      update_scale=20.0, n_malicious=2)
    received = [np.zeros((4,), dtype=np.float32)]
    returned = [np.ones((4,), dtype=np.float32)]
    out = apply_attack(returned, received, cid=0, spec=spec)
    assert np.allclose(out[0], 20.0)
    # honest client untouched
    honest = apply_attack(returned, received, cid=5, spec=spec)
    assert np.allclose(honest[0], 1.0)


def test_apply_attack_replacement_is_large_and_seeded():
    spec = AttackSpec(enabled=True, attack_type="replacement",
                      update_scale=10.0, n_malicious=2)
    received = [np.zeros((4,), dtype=np.float32)]
    returned = [np.ones((4,), dtype=np.float32)]
    out = apply_attack(returned, received, cid=0, spec=spec)
    assert np.linalg.norm(out[0]) > 50.0  # scale=100 standard normal
    out2 = apply_attack(returned, received, cid=0, spec=spec)
    assert np.array_equal(out[0], out2[0])


def test_attack_report_notes_synthetic():
    spec = AttackSpec(enabled=True, attack_type="label_flip", n_malicious=1)
    from src.federated.defense.attack import attack_report
    rep = attack_report(spec, [0])
    assert rep["n_malicious"] == 1
    assert "synthetic" in rep["note"]


# ----------------------------------------------------------------------
# clipping
# ----------------------------------------------------------------------

def test_clipper_no_threshold_passthrough():
    clipper = UpdateClipper()
    u = np.full(10, 3.0)
    out = clipper.clip(u)
    assert np.array_equal(out, u)
    s = clipper.summary()
    assert s["n_updates"] == 1 and s["n_clipped"] == 0


def test_clipper_clips_and_records():
    clipper = UpdateClipper(max_norm=1.0)
    u = np.full(10, 3.0)  # norm = 3*sqrt(10) ≈ 9.49
    out = clipper.clip(u)
    assert np.linalg.norm(out) == pytest.approx(1.0)
    rec = clipper.records[-1]
    assert rec.clipped
    assert rec.norm_before == pytest.approx(np.linalg.norm(u))
    assert rec.threshold == 1.0
    assert rec.norm_after == pytest.approx(1.0)
    assert rec.scale == pytest.approx(1.0 / np.linalg.norm(u))


def test_clipper_under_threshold_untouched():
    clipper = UpdateClipper(max_norm=10.0)
    u = np.full(10, 1.0)
    out = clipper.clip(u)
    assert np.array_equal(out, u)
    assert clipper.records[-1].clipped is False
    assert clipper.records[-1].scale == 1.0


def test_clipper_rejects_bad_threshold():
    with pytest.raises(ValueError):
        UpdateClipper(max_norm=0.0)
    with pytest.raises(ValueError):
        UpdateClipper(max_norm=-1.0)


# ----------------------------------------------------------------------
# anomaly detection
# ----------------------------------------------------------------------

def test_anomaly_normal_vs_outlier():
    det = UpdateAnomalyDetector(suspect_mult=3.0, detect_mult=6.0)
    updates = [(str(i), np.full(64, 0.01, dtype=np.float32) * (i + 1))
               for i in range(8)]
    updates[7] = ("7", np.full(64, 1.0, dtype=np.float32))  # big outlier
    records = det.score_and_classify(updates)
    by_cid = {r.cid: r.classification for r in records}
    assert by_cid["7"] in (AnomalyClass.SUSPICIOUS, AnomalyClass.HIGHLY_ANOMALOUS)
    for i in range(7):
        assert by_cid[str(i)] == AnomalyClass.NORMAL
    s = det.summary()
    assert s["n_updates"] == 8
    assert s["n_highly_anomalous"] + s["n_suspicious"] >= 1


def test_anomaly_reference_distance():
    det = UpdateAnomalyDetector(suspect_mult=2.0, detect_mult=4.0)
    # updates clustered around 0.1 except one far outlier
    updates = [(str(i), np.full(32, 0.01 + 0.002 * i, dtype=np.float32))
               for i in range(5)]
    updates.append(("5", np.full(32, 2.0, dtype=np.float32)))
    records = det.score_and_classify(updates, reference_norm=0.1)
    by_cid = {r.cid: r.classification for r in records}
    assert by_cid["5"] in (AnomalyClass.SUSPICIOUS, AnomalyClass.HIGHLY_ANOMALOUS)
    for i in range(5):
        assert by_cid[str(i)] == AnomalyClass.NORMAL


def test_anomaly_flags_exact_cids():
    det = UpdateAnomalyDetector(suspect_mult=3.0, detect_mult=6.0)
    updates = [(str(i), np.full(32, 0.01, dtype=np.float32) * (i + 1))
               for i in range(5)]
    updates.append(("5", np.full(32, 5.0, dtype=np.float32)))
    det.score_and_classify(updates)
    assert "5" in det.flagged_cids()
    assert all(c not in det.flagged_cids() for c in ("0", "1", "2", "3", "4"))


def test_anomaly_bad_thresholds():
    with pytest.raises(ValueError):
        UpdateAnomalyDetector(suspect_mult=5.0, detect_mult=2.0)


# ----------------------------------------------------------------------
# robust aggregation
# ----------------------------------------------------------------------

def test_coordinate_wise_median():
    updates = [np.array([1.0, 2.0]), np.array([3.0, 4.0]),
               np.array([5.0, 6.0]), np.array([7.0, 8.0]),
               np.array([100.0, 0.0])]  # outlier at coord 0
    med = coordinate_wise_median(updates)
    assert med[0] == pytest.approx(5.0)
    assert med[1] == pytest.approx(4.0)


def test_coordinate_wise_median_empty():
    with pytest.raises(ValueError):
        coordinate_wise_median([])


def test_trimmed_mean_removes_outliers():
    updates = [np.array([1.0]), np.array([2.0]), np.array([3.0]),
               np.array([4.0]), np.array([5.0]), np.array([100.0])]
    tm = trimmed_mean(updates, trim_frac=0.2)
    # trims 1 largest + 1 smallest → mean(2,3,4,5) = 3.5
    assert tm[0] == pytest.approx(3.5)


def test_trimmed_mean_zero_trim_is_mean():
    updates = [np.array([1.0]), np.array([3.0])]
    assert trimmed_mean(updates, trim_frac=0.0)[0] == pytest.approx(2.0)


def test_trimmed_mean_bad_trim():
    with pytest.raises(ValueError):
        trimmed_mean([np.array([1.0])], trim_frac=0.7)


# ----------------------------------------------------------------------
# validation gate
# ----------------------------------------------------------------------

class _FakeModelBuilder:
    def __init__(self, f1_by_call):
        self.f1_by_call = list(f1_by_call)
        self.calls = 0

    def f1(self, parameters):
        f1 = self.f1_by_call[self.calls % len(self.f1_by_call)]
        self.calls += 1
        return f1


class _FakeGate(ValidationGate):
    def __init__(self, f1s, tolerance=0.01, flag_delta=0.02):
        self._fake = _FakeModelBuilder(f1s)
        self.tolerance = tolerance
        self.flag_delta = flag_delta
        self.trusted_params = None
        self.trusted_f1 = None
        self.records = []
        self.n_rejects = 0
        self.n_flags = 0

    def _evaluate_f1(self, parameters):
        return self._fake.f1(parameters)


def test_validation_gate_accepts_first():
    gate = _FakeGate([0.90])
    rec = gate.validate(1, [np.array([1.0])])
    assert rec.decision == ValidationDecision.ACCEPT
    assert gate.trusted_f1 == pytest.approx(0.90)


def test_validation_gate_rejects_degradation():
    gate = _FakeGate([0.90, 0.80])
    gate.validate(1, [np.array([1.0])])
    rec = gate.validate(2, [np.array([2.0])])
    assert rec.decision == ValidationDecision.REJECT
    assert gate.n_rejects == 1
    # trusted model retained
    assert gate.trusted_f1 == pytest.approx(0.90)


def test_validation_gate_flags_small_drop():
    gate = _FakeGate([0.90, 0.89])
    gate.validate(1, [np.array([1.0])])
    rec = gate.validate(2, [np.array([2.0])])
    assert rec.decision == ValidationDecision.FLAG
    assert gate.n_flags == 1


def test_validation_gate_accepts_improvement():
    gate = _FakeGate([0.90, 0.92])
    gate.validate(1, [np.array([1.0])])
    rec = gate.validate(2, [np.array([2.0])])
    assert rec.decision == ValidationDecision.ACCEPT
    assert gate.trusted_f1 == pytest.approx(0.92)


def test_validation_gate_summary():
    gate = _FakeGate([0.90, 0.80])
    gate.validate(1, [np.array([1.0])])
    gate.validate(2, [np.array([2.0])])
    s = gate.summary()
    assert s["n_validations"] == 2
    assert s["n_accept"] == 1
    assert s["n_reject"] == 1
    assert s["trusted_f1"] == pytest.approx(0.90)


# ----------------------------------------------------------------------
# strategy-level defense pipeline
# ----------------------------------------------------------------------

def _fake_fitres(params_ndarrays, num_examples=100):
    from flwr.common import ndarrays_to_parameters
    return type("FitRes", (), {
        "parameters": ndarrays_to_parameters(params_ndarrays),
        "num_examples": num_examples,
        "metrics": {"upload_bytes": 100, "download_bytes": 100},
    })()


def _fake_client_proxy(cid: str):
    return type("ClientProxy", (), {"cid": cid})()


def test_defended_tracked_none_mode_matches_fedavg():
    from src.federated.fl.strategy import FedAvgTracked
    n_clients, rounds = 4, 1
    kwargs = dict(
        num_clients=n_clients, fraction_fit=1.0, num_rounds=rounds,
        local_epochs=1, learning_rate=1e-3, batch_size=64,
        evaluate_every=1, initial_parameters=_params(8),
        evaluate_fn=None, fraction_evaluate=1.0,
    )
    plain = FedAvgTracked(**kwargs)
    defended = DefendedTracked(defense_mode="none", **kwargs)
    results = [
        (_fake_client_proxy(str(i)),
         _fake_fitres([np.full(8, 1.0 + 0.1 * i, dtype=np.float32)]))
        for i in range(n_clients)
    ]
    p1, _ = plain.aggregate_fit(1, results, [])
    p2, _ = defended.aggregate_fit(1, results, [])
    from flwr.common import parameters_to_ndarrays
    a = parameters_to_ndarrays(p1)
    b = parameters_to_ndarrays(p2)
    assert all(np.allclose(x, y, atol=1e-6) for x, y in zip(a, b))
    assert len(defended.rounds) == 1
    s = defended.summarize_defense()
    assert s["n_rounds"] == 1


def _params(size=4):
    from flwr.common import ndarrays_to_parameters
    return ndarrays_to_parameters([np.zeros(size, dtype=np.float32)])


def test_defended_tracked_anomaly_excludes_outlier():
    kwargs = dict(
        num_clients=5, fraction_fit=1.0, num_rounds=1,
        local_epochs=1, learning_rate=1e-3, batch_size=64,
        evaluate_every=1, initial_parameters=_params(16),
        evaluate_fn=None, fraction_evaluate=1.0,
    )
    strategy = DefendedTracked(defense_mode="anomaly",
                               anomaly_suspect_mult=3.0,
                               anomaly_detect_mult=6.0, **kwargs)
    results = []
    for i in range(4):
        results.append((_fake_client_proxy(str(i)),
                        _fake_fitres([np.full(16, 0.01 * (i + 1),
                                              dtype=np.float32)])))
    results.append((_fake_client_proxy("4"),
                    _fake_fitres([np.full(16, 10.0, dtype=np.float32)])))
    agg, _ = strategy.aggregate_fit(1, results, [])
    rec = strategy.rounds[0]
    assert rec["n_excluded_anomalous"] == 1
    assert rec["excluded_cids"] == ["4"]
    assert rec["n_clients_aggregated"] == 4
    from flwr.common import parameters_to_ndarrays
    out = parameters_to_ndarrays(agg)[0]
    # outlier (10.0) must NOT dominate: |agg| stays small
    assert np.abs(out).max() < 1.0


def test_defended_tracked_validation_rejects_keeps_trusted():
    from src.federated.defense.validation import ValidationDecision
    kwargs = dict(
        num_clients=3, fraction_fit=1.0, num_rounds=2,
        local_epochs=1, learning_rate=1e-3, batch_size=64,
        evaluate_every=1, initial_parameters=_params(8),
        evaluate_fn=None, fraction_evaluate=1.0,
    )
    gate = _FakeGate([0.95, 0.70])  # second candidate degrades
    strategy = DefendedTracked(defense_mode="validation",
                               validation_gate=gate, **kwargs)
    for rnd in (1, 2):
        results = [(_fake_client_proxy(str(i)),
                    _fake_fitres([np.full(8, 0.1 * rnd, dtype=np.float32)]))
                   for i in range(3)]
        strategy.aggregate_fit(rnd, results, [])
    rec2 = strategy.rounds[1]
    assert rec2["validation"]["decision"] == "REJECT"
    assert strategy.summarize_defense()["validation"]["n_reject"] == 1


def test_defended_tracked_clipping_mode_records():
    kwargs = dict(
        num_clients=3, fraction_fit=1.0, num_rounds=1,
        local_epochs=1, learning_rate=1e-3, batch_size=64,
        evaluate_every=1, initial_parameters=_params(8),
        evaluate_fn=None, fraction_evaluate=1.0,
    )
    strategy = DefendedTracked(defense_mode="clipping", clip_norm=0.5,
                               **kwargs)
    results = [(_fake_client_proxy(str(i)),
                _fake_fitres([np.full(8, 2.0, dtype=np.float32)]))
               for i in range(3)]
    strategy.aggregate_fit(1, results, [])
    rec = strategy.rounds[0]
    clips = rec["clipping"]
    assert len(clips) == 3
    assert all(c["clipped"] for c in clips)
    assert all(c["threshold"] == 0.5 for c in clips)
    s = strategy.summarize_defense()
    assert s["clipping"]["n_clipped"] == 3


def test_defended_tracked_bad_mode():
    kwargs = dict(
        num_clients=2, fraction_fit=1.0, num_rounds=1,
        local_epochs=1, learning_rate=1e-3, batch_size=64,
        evaluate_every=1, initial_parameters=_params(8),
        evaluate_fn=None, fraction_evaluate=1.0,
    )
    with pytest.raises(ValueError):
        DefendedTracked(defense_mode="nonsense", **kwargs)


def test_defended_tracked_multilayer_aggregation():
    """Multi-layer models: the flattened update must be split back per layer."""
    from flwr.common import ndarrays_to_parameters
    init = [np.zeros((4, 3), dtype=np.float32), np.zeros(2, dtype=np.float32)]
    kwargs = dict(
        num_clients=3, fraction_fit=1.0, num_rounds=1,
        local_epochs=1, learning_rate=1e-3, batch_size=64,
        evaluate_every=1, initial_parameters=ndarrays_to_parameters(init),
        evaluate_fn=None, fraction_evaluate=1.0,
    )
    for mode in ("clipping", "anomaly", "validation", "robust_median",
                 "robust_trimmed"):
        gate = _FakeGate([0.9]) if mode == "validation" else None
        strategy = DefendedTracked(
            defense_mode=mode, clip_norm=1.0, validation_gate=gate, **kwargs)
        results = [
            (_fake_client_proxy(str(i)),
             _fake_fitres([np.full((4, 3), 0.5 * (i + 1), dtype=np.float32),
                           np.full(2, 0.1 * (i + 1), dtype=np.float32)]))
            for i in range(3)
        ]
        agg, _ = strategy.aggregate_fit(1, results, [])
        from flwr.common import parameters_to_ndarrays
        layers = parameters_to_ndarrays(agg)
        assert len(layers) == 2
        assert layers[0].shape == (4, 3)
        assert layers[1].shape == (2,)
        assert np.isfinite(layers[0]).all()


# ----------------------------------------------------------------------
# config wiring
# ----------------------------------------------------------------------

def test_attack_defense_config_defaults():
    cfg = ExperimentConfig()
    assert cfg.attack.enabled is False
    assert cfg.attack.attack_type == "none"
    assert cfg.defense.mode == "none"
    assert cfg.defense.clip_norm is None
    assert cfg.defense.validation_tolerance == 0.01


def test_attack_spec_from_config():
    spec = AttackSpec.from_config(AttackConfig(enabled=True,
                                               attack_type="label_flip",
                                               n_malicious=3))
    assert spec.is_malicious
    assert spec.n_malicious == 3
    assert spec.attack_type == "label_flip"


def test_defense_config_in_yaml_roundtrip(tmp_path):
    cfg = ExperimentConfig()
    cfg.attack.enabled = True
    cfg.attack.attack_type = "scaled_update"
    cfg.defense.mode = "clipping"
    cfg.defense.clip_norm = 0.1
    d = cfg.to_dict()
    assert d["attack"]["enabled"] is True
    assert d["defense"]["mode"] == "clipping"
    rebuilt = ExperimentConfig.from_dict(d)
    assert rebuilt.attack.attack_type == "scaled_update"
    assert rebuilt.defense.clip_norm == 0.1