"""Phase 9: FedAvg over saved partitions (synthetic data, tiny config).

Proves the FL machinery works end-to-end with a REAL Flower server (local
gRPC) + NumPyClient workers over a REAL saved Phase-4 partition:
- partition config is consumed, never regenerated inside FL
- clients receive the global model, train locally, return params/metrics
- per-round global eval + avg/worst client F1 + communication bytes recorded
- results are JSON-serializable and match the tracked strategy history
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedshield.config import ExperimentConfig  # noqa: E402
from src.federated.data.partition import (  # noqa: E402
    ClientPartitionConfig, build_client_partition, save_partition,
)
from src.federated.fl.dataset import PartitionClientData  # noqa: E402
from src.federated.fl.server import run_fedavg_experiment  # noqa: E402


def make_synthetic(n=1800, n_test=400, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 16)).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] + 0.5 * rng.normal(size=n)) > 0).astype(np.int64)
    X_test = rng.normal(size=(n_test, 16)).astype(np.float32)
    y_test = ((X_test[:, 0] + X_test[:, 1] + 0.5 * rng.normal(size=n_test)) > 0).astype(np.int64)
    return X, y, X_test, y_test


@pytest.fixture(scope="module")
def fl_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("fl")
    X, y, X_test, y_test = make_synthetic()

    pdir = base / "iid-c3-s7"
    pcfg = ClientPartitionConfig(strategy="iid", clients=3, seed=7,
                                 val_fraction=0.1, min_samples_per_client=0)
    part = build_client_partition(y, pcfg)
    save_partition(pdir, part, y)

    cfg = ExperimentConfig()
    cfg.fl.num_clients = 3
    cfg.fl.num_rounds = 2
    cfg.fl.client_fraction = 1.0
    cfg.train.local_epochs = 1
    cfg.train.learning_rate = 0.01
    cfg.train.batch_size = 64
    cfg.model.input_dim = 16
    cfg.model.hidden_layers = (8, 4)
    cfg.model.dropout = 0.0
    cfg.seed = 42

    out = base / "results"
    scale_inv = np.ones(16, dtype=np.float32)
    results = run_fedavg_experiment(cfg, pdir, X, y, X_test, y_test,
                                    scale_inv=scale_inv, output_dir=out, seed=cfg.seed)
    return {"cfg": cfg, "pdir": pdir, "out": out, "results": results,
            "X": X, "y": y}


def test_partition_consumed_not_regenerated(fl_env):
    """FL uses the saved partition; client sizes must match the manifest."""
    data = PartitionClientData(fl_env["pdir"], fl_env["X"], fl_env["y"])
    assert data.n_clients == 3
    assert data.strategy() == "iid"
    sizes = data.client_sizes()
    assert all(tr + va > 0 for tr, va in sizes.values())
    assert data.config.seed == 7 and data.config.clients == 3


def test_client_data_is_disjoint(fl_env):
    data = PartitionClientData(fl_env["pdir"], fl_env["X"], fl_env["y"])
    sets = [set(data.train_idx[c].tolist()) | set(data.val_idx[c].tolist())
            for c in range(3)]
    assert all(sets[i].isdisjoint(sets[j])
               for i in range(3) for j in range(i + 1, 3))


def test_run_completes_with_two_rounds(fl_env):
    res = fl_env["results"]
    assert len(res["rounds"]) == 2
    for r in res["rounds"]:
        assert r["round"] in (1, 2)
        assert r["round_time_ms"] > 0
        assert r["n_clients_fit"] == 3


def test_global_metrics_recorded(fl_env):
    for r in fl_env["results"]["rounds"]:
        ge = r["global_eval"]
        assert ge is not None
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "n_test"):
            assert k in ge
        assert ge["roc_auc"] is not None
        assert ge["n_test"] == 400


def test_avg_and_worst_client_f1(fl_env):
    for r in fl_env["results"]["rounds"]:
        assert r["avg_client_f1"] is not None
        assert r["worst_client_f1"] is not None
        assert r["worst_client_f1"] <= r["avg_client_f1"] <= 1.0
    pc = fl_env["results"]["per_client_metrics"]
    assert len(pc) == 2
    assert len(pc[0]["clients"]) == 3
    for c in pc[0]["clients"]:
        assert {"cid", "accuracy", "f1", "roc_auc", "n_val"} <= set(c)


def test_communication_is_measured(fl_env):
    comm = fl_env["results"]["communication"]
    assert comm["model_parameter_bytes"] > 0
    rounds = comm["per_round"]
    assert len(rounds) == 2
    for r in rounds:
        assert r["download_bytes"] > 0
        assert r["upload_bytes"] > 0
        assert r["bytes_this_round"] == r["download_bytes"] + r["upload_bytes"]
    tot = comm["totals"]
    assert tot["total_upload_bytes"] == sum(r["upload_bytes"] for r in rounds)
    assert tot["total_download_bytes"] == sum(r["download_bytes"] for r in rounds)
    # 3 clients x 2 rounds, each exchanging model_parameter_bytes
    assert tot["total_upload_bytes"] == 3 * 2 * comm["model_parameter_bytes"]


def test_results_are_honest_and_serializable(fl_env):
    res = fl_env["results"]
    payload = json.dumps(res)
    assert "fedavg" in payload
    assert res["training_time_s"] > 0
    assert res["round_time_s_mean"] > 0
    assert res["final_global_test_metrics"] is not None


def test_output_files_written(fl_env):
    out = fl_env["out"]
    assert (out / "config.json").exists()
    assert (out / "rounds.jsonl").exists()
    assert (out / "summary.json").exists()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment"]["algorithm"] == "fedavg"
    assert summary["experiment"]["strategy"] == "iid"
    assert summary["communication"]["totals"]["total_bytes_exchanged"] > 0


def test_fedavg_improves_or_stays_correct(fl_env):
    """FedAvg must at least produce sensible metrics on a learnable task."""
    res = fl_env["results"]
    f0 = res["rounds"][0]["global_eval"]
    f1 = res["rounds"][-1]["global_eval"]
    assert f1["accuracy"] >= f0["accuracy"] - 0.05  # no catastrophic divergence
    assert f1["roc_auc"] > 0.5


def test_mismatched_clients_rejected(tmp_path_factory):
    X, y, X_test, y_test = make_synthetic(n=600)
    pdir = tmp_path_factory.mktemp("fl2") / "iid-c3-s7"
    save_partition(pdir, build_client_partition(y, ClientPartitionConfig(
        strategy="iid", clients=3, seed=7, min_samples_per_client=0)), y)
    cfg = ExperimentConfig()
    cfg.fl.num_clients = 4  # does not match the 3-client partition
    with pytest.raises(ValueError, match="partition has 3 clients"):
        run_fedavg_experiment(cfg, pdir, X, y, X_test, y_test)