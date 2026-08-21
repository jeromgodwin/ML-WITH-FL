"""Tests for Phase 16 communication efficiency (actual bytes, no estimates)."""

import json
import pathlib


def test_actual_bytes_measurement():
    """Model bytes must be actual p.nbytes sum, not estimate — check serialize_bytes."""
    from src.federated.fl.client import serialize_bytes
    import numpy as np

    params = [np.zeros((10, 10), dtype=np.float32), np.zeros((5,), dtype=np.float32)]
    expected = 10 * 10 * 4 + 5 * 4
    assert serialize_bytes(params) == expected
    # Per-round upload/download are summed from client metrics (strategy.py:95)
    # This is an actual measurement via p.nbytes on real ndarrays, not an estimate.


def test_communication_record_no_invention():
    """CommunicationRecord must leave missing values as None, not invent."""
    from src.federated.communication.analysis import from_summary

    # Summary with missing communication should not invent bytes
    summary = {
        "experiment": {"algorithm": "fedavg", "strategy": "iid", "fl_config": {"num_clients": 10, "num_rounds": 5}},
        "final_global_test_metrics": {"f1": 0.9},
        "training_time_s": 100.0,
        # No communication block — must stay None
    }
    rec = from_summary(summary, "test-id")
    assert rec.total_bytes is None
    assert rec.bytes_per_round is None
    assert rec.model_parameter_count is None
    assert rec.final_f1 == 0.9


def test_tradeoffs_filter_missing():
    """Tradeoff pairs only include experiments where both axes are present."""
    from src.federated.communication.analysis import CommunicationRecord, compute_tradeoffs

    records = [
        CommunicationRecord(experiment_id="a", algorithm="fedavg", total_bytes=1000, final_f1=0.9, training_time_s=10.0, n_clients=10, convergence_round=2),
        CommunicationRecord(experiment_id="b", algorithm="fedavg", total_bytes=None, final_f1=0.8, training_time_s=20.0, n_clients=10),
        CommunicationRecord(experiment_id="c", algorithm="fedavg", total_bytes=2000, final_f1=None, training_time_s=15.0, n_clients=10),
    ]
    t = compute_tradeoffs(records)
    # Only 'a' has both total_bytes and final_f1
    assert len(t["communication_vs_f1"]) == 1
    assert t["communication_vs_f1"][0]["experiment_id"] == "a"
    # vs training_time: a and c have both (b missing bytes)
    assert len(t["communication_vs_training_time"]) == 2


def test_compare_algorithms_groups():
    """compare_algorithms must group by algorithm / strategy / rounds / clients."""
    from src.federated.communication.analysis import CommunicationRecord, compare_algorithms

    records = [
        CommunicationRecord(experiment_id="1", algorithm="fedavg", strategy="iid", n_rounds=5, n_clients=10, final_f1=0.9, total_bytes=1000),
        CommunicationRecord(experiment_id="2", algorithm="fedprox", strategy="iid", n_rounds=5, n_clients=10, final_f1=0.91, total_bytes=1000),
        CommunicationRecord(experiment_id="3", algorithm="personalized", strategy="severe", n_rounds=10, n_clients=10, final_f1=0.85, total_bytes=2000),
    ]
    cmp = compare_algorithms(records)
    assert "fedavg" in cmp["by_algorithm"]
    assert "iid" in cmp["by_strategy"]
    assert "5" in cmp["by_rounds"]
    assert "10" in cmp["by_clients"]
    assert cmp["by_algorithm"]["fedavg"]["n_experiments"] == 1
    assert cmp["by_rounds"]["5"]["mean_f1"] is not None


def test_phase16_server_measures_actual_bytes(tmp_path):
    """Ensure server's communication block contains actual measured fields when run is mocked.

    We test the analyzer's from_summary handling of the new fields:
    model_parameter_count, model_parameter_bytes, per_round with n_clients.
    """
    from src.federated.communication.analysis import from_summary

    summary = {
        "experiment": {
            "algorithm": "personalized",
            "strategy": "iid",
            "fl_config": {"num_clients": 10, "num_rounds": 5, "client_fraction": 1.0, "proximal_mu": 0.0},
        },
        "communication": {
            "model_parameter_count": 642688,
            "model_parameter_bytes": 2570752,
            "full_model_parameter_count": 642817,
            "full_model_bytes": 2571268,
            "per_round": [
                {"round": 1, "download_bytes": 25707520, "upload_bytes": 25707520, "bytes_this_round": 51415040, "n_clients": 10},
                {"round": 2, "download_bytes": 25707520, "upload_bytes": 25707520, "bytes_this_round": 51415040, "n_clients": 10},
            ],
            "totals": {
                "total_bytes_exchanged": 102830080,
                "total_download_bytes": 51415040,
                "total_upload_bytes": 51415040,
                "rounds": 2,
            },
        },
        "final_global_test_metrics": {"f1": 0.88, "roc_auc": 0.96},
        "training_time_s": 200.0,
        "rounds": [
            {"round": 1, "global_eval": {"f1": 0.8}},
            {"round": 2, "global_eval": {"f1": 0.88}},
        ],
    }
    rec = from_summary(summary, "pers-001")
    assert rec.model_parameter_count == 642688
    assert rec.model_parameter_bytes == 2570752
    # Personalized body is smaller than full model
    assert rec.model_parameter_count < rec.full_model_parameter_count
    assert rec.total_bytes == 102830080
    assert rec.bytes_per_round == 102830080 / 5  # total / n_rounds (fl_config)
    assert rec.convergence_round == 2  # reaches 95% of 0.88 at round 2
