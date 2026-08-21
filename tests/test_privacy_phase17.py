"""Tests for Phase 17 privacy and privacy-utility analysis."""

import pathlib


def test_privacy_invariants():
    """Raw files/rows stay local; server receives only model updates."""
    from src.federated.privacy.verification import verify_privacy_invariants

    inv = verify_privacy_invariants()
    assert inv["raw_files_remain_local"] is True
    assert inv["raw_feature_rows_remain_local"] is True
    assert inv["server_receives_only_updates"] is True
    assert inv["summary"] == "VERIFIED"


def test_dp_clipping_where():
    """Clipping occurs client-side on update vector with max_grad_norm."""
    from src.federated.privacy.dp import clip_update
    import numpy as np

    upd = np.array([3.0, 4.0], dtype=np.float32)  # norm 5
    clipped, did_clip, orig, clipped_norm = clip_update(upd, max_norm=2.5)
    assert did_clip is True
    assert abs(orig - 5.0) < 1e-5
    assert abs(clipped_norm - 2.5) < 1e-5
    # Clipped vector is scaled
    assert abs(float(np.linalg.norm(clipped)) - 2.5) < 1e-5
    # No clip when under bound
    upd2 = np.array([0.5, 0.5], dtype=np.float32)
    _, did_clip2, _, _ = clip_update(upd2, max_norm=1.0)
    assert did_clip2 is False


def test_dp_noise_where():
    """Noise is Gaussian after clipping, per-coordinate, same shape."""
    from src.federated.privacy.dp import add_gaussian_noise
    import numpy as np

    upd = np.zeros((10,), dtype=np.float32)
    rng = np.random.default_rng(0)
    noised = add_gaussian_noise(upd, sigma=1.0, rng=rng)
    assert noised.shape == upd.shape
    assert not np.allclose(noised, upd)  # noise added
    # sigma 0 means no noise
    rng2 = np.random.default_rng(0)
    noised2 = add_gaussian_noise(upd, sigma=0.0, rng=rng2)
    assert np.allclose(noised2, upd)


def test_dp_parameters_and_assumptions_documented():
    """privacy_report must document where clipping/noise occur and assumptions."""
    from src.federated.privacy.dp import PrivacySpec, privacy_report

    spec = PrivacySpec(enabled=True, noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5)
    rep = privacy_report(spec, rounds=5, sampling_rate=1.0)
    assert rep["enabled"] is True
    assert "where_clipping" in rep
    assert "client-side" in rep["where_clipping"]
    assert "where_noise" in rep
    assert "client-side" in rep["where_noise"]
    assert "parameters" in rep
    assert rep["parameters"]["max_grad_norm (C)"] == 1.0
    assert rep["parameters"]["noise_multiplier (sigma/C)"] == 1.0
    assert "assumptions" in rep and len(rep["assumptions"]) >= 3
    assert rep["epsilon_estimate"] is not None
    # No DP case
    spec2 = PrivacySpec(enabled=False)
    rep2 = privacy_report(spec2, rounds=5)
    assert rep2["enabled"] is False


def test_privacy_strength_settings():
    """Moderate vs stronger privacy have different noise_multiplier/sigma."""
    from src.federated.privacy.dp import PrivacySpec

    moderate = PrivacySpec(enabled=True, noise_multiplier=1.0, max_grad_norm=1.0)
    stronger = PrivacySpec(enabled=True, noise_multiplier=2.0, max_grad_norm=1.0)
    assert moderate.sigma == 1.0
    assert stronger.sigma == 2.0
    assert moderate.strength_label() == "moderate"
    assert stronger.strength_label() == "strong"


def test_privacy_utility_analysis_no_invention():
    """Privacy-utility table must not invent missing values."""
    from src.federated.privacy.analysis import privacy_utility_table

    # No DP baseline missing F1 stays None, not invented
    summaries = [
        {"final_global_test_metrics": {"f1": 0.9, "accuracy": 0.92}, "communication": {"totals": {"total_bytes_exchanged": 1000}}, "training_time_s": 10.0, "experiment": {"fl_config": {"num_rounds": 5}}, "privacy": {"enabled": False}},
        {"final_global_test_metrics": {"f1": None, "accuracy": None}, "communication": {"totals": {"total_bytes_exchanged": 1000}}, "training_time_s": 12.0, "experiment": {"fl_config": {"num_rounds": 5}}, "privacy": {"enabled": True, "config": {"sigma": 1.0, "noise_multiplier": 1.0, "max_grad_norm": 1.0, "delta": 1e-5}}},
    ]
    rows = privacy_utility_table(summaries, ["no_dp", "moderate"])
    assert rows[0]["f1"] == 0.9
    assert rows[1]["f1"] is None  # not invented
    assert rows[1]["sigma"] == 1.0


def test_client_dp_integration():
    """FedAvgClient should apply DP when privacy_spec enabled (even with tiny data, check metrics)."""
    import numpy as np
    from src.federated.models.mlp import MLPConfig
    from src.federated.fl.client import FedAvgClient
    from src.federated.privacy.dp import PrivacySpec

    cfg = MLPConfig(input_dim=4, hidden_layers=(8,), dropout=0.0)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = (rng.random(20) > 0.5).astype(np.int64)
    spec = PrivacySpec(enabled=True, noise_multiplier=1.0, max_grad_norm=0.5, seed=123)
    client = FedAvgClient(cfg, X[:15], y[:15], X[15:], y[15:], seed=42, privacy_spec=spec)
    # Get initial params
    init_params = client.get_parameters({})
    params, n, metrics = client.fit(init_params, {"server_round": 0, "local_epochs": 1, "lr": 1e-3, "batch_size": 8})
    # DP metrics should be present
    assert metrics.get("dp_applied") is True
    assert "dp_sigma" in metrics
    assert "dp_did_clip" in metrics
    # Upload bytes should be actual size (same as download)
    assert metrics["upload_bytes"] == metrics["download_bytes"]
