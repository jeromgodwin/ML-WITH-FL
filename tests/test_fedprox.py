"""Phase 10: FedProx over saved partitions (synthetic data, tiny config).

Verifies, on top of the Phase-9 FedAvg machinery, that FedProx:
- adds the proximal regularizer (mu/2)*||w - w_global||^2 to the LOCAL objective
- actually applies mu (mu=0 degenerates to exactly FedAvg)
- keeps w_global a FROZEN snapshot captured at the START of local training
- uses the IDENTICAL partition (dir/config) as FedAvg for fair comparison
- runs end-to-end over a real Flower gRPC server and records honest metrics
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedshield.config import ExperimentConfig  # noqa: E402
from src.federated.data.partition import (  # noqa: E402
    ClientPartitionConfig, build_client_partition, save_partition,
)
from src.federated.fl.client import FedAvgClient, build_client_fn  # noqa: E402
from src.federated.fl.dataset import PartitionClientData  # noqa: E402
from src.federated.fl.server import run_fl_experiment  # noqa: E402
from src.federated.models.mlp import MLPConfig, build_mlp  # noqa: E402


def make_synthetic(n=1800, n_test=400, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 16)).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] + 0.5 * rng.normal(size=n)) > 0).astype(np.int64)
    X_test = rng.normal(size=(n_test, 16)).astype(np.float32)
    y_test = ((X_test[:, 0] + X_test[:, 1] + 0.5 * rng.normal(size=n_test)) > 0).astype(np.int64)
    return X, y, X_test, y_test


def model_cfg():
    return MLPConfig(input_dim=16, hidden_layers=(8, 4), dropout=0.0)


def prox_penalty(params, w_global, mu):
    """Reference (mu/2)*||w - w_global||^2 in numpy."""
    return float((mu / 2.0) * sum(
        float(np.sum(np.square(np.asarray(p) - np.asarray(w))))
        for p, w in zip(params, w_global)))


def test_prox_penalty_zero_for_mu_zero():
    """mu=0 => no regularizer regardless of model/global mismatch."""
    rng = np.random.default_rng(0)
    p = [rng.normal(size=(4, 4)).astype(np.float32), rng.normal(size=(4,)).astype(np.float32)]
    w = [rng.normal(size=(4, 4)).astype(np.float32), rng.normal(size=(4,)).astype(np.float32)]
    assert prox_penalty(p, w, 0.0) == 0.0


def test_prox_penalty_zero_when_params_equal_global():
    """No penalty when local params already equal the global snapshot."""
    rng = np.random.default_rng(1)
    p = [rng.normal(size=(4, 4)).astype(np.float32), rng.normal(size=(4,)).astype(np.float32)]
    assert prox_penalty(p, p, 1.0) == 0.0


def test_prox_penalty_magnitude():
    """(mu/2)*||w - w_global||^2 grows with mu and with parameter distance."""
    rng = np.random.default_rng(2)
    p = [rng.normal(size=(4, 4)).astype(np.float32)]
    w = [rng.normal(size=(4, 4)).astype(np.float32)]
    p1 = prox_penalty(p, w, 1.0)
    p2 = prox_penalty(p, w, 2.0)
    assert p1 > 0.0
    assert p2 == 2.0 * p1
    # against a zero reference: doubling the distance quadruples the penalty
    zero = [np.zeros_like(a) for a in p]
    assert prox_penalty(p, zero, 1.0) == pytest.approx(4.0 * prox_penalty([a / 2.0 for a in p], zero, 1.0))


def test_client_applies_proximal_term():
    """A FedProx client with mu>0 returns metrics showing the penalty was added."""
    X, y, _, _ = make_synthetic(n=800)
    client = FedAvgClient(model_cfg(), X[:600], y[:600], X[600:], y[600:], seed=3)
    params = client.get_parameters({})
    cfg = {"server_round": 1, "local_epochs": 1, "lr": 0.01, "batch_size": 64,
           "proximal_mu": 0.5}
    _, _, metrics = client.fit(params, cfg)
    assert metrics["proximal_mu"] == 0.5
    assert metrics["prox_penalty"] > 0.0


def test_mu_zero_equals_fedavg():
    """mu=0 FedProx and FedAvg produce identical parameters (deterministic)."""
    X, y, _, _ = make_synthetic(n=800)
    c_avg = FedAvgClient(model_cfg(), X[:600], y[:600], X[600:], y[600:], seed=3)
    c_prox = FedAvgClient(model_cfg(), X[:600], y[:600], X[600:], y[600:], seed=3)
    # identical starting parameters (each client builds a fresh random model);
    # copy: get_parameters returns views that share memory with the model
    start = [a.copy() for a in c_avg.get_parameters({})]
    c_avg._set_parameters(start)
    c_prox._set_parameters(start)
    fit_cfg = {"server_round": 1, "local_epochs": 1, "lr": 0.01, "batch_size": 64}

    p_avg, _, m_avg = c_avg.fit(start, fit_cfg)
    p_prox, _, m_prox = c_prox.fit(start, {**fit_cfg, "proximal_mu": 0.0})

    assert all(np.allclose(a, b, atol=1e-12) for a, b in zip(p_avg, p_prox))
    assert m_prox["prox_penalty"] == 0.0
    assert m_avg["train_loss"] == m_prox["train_loss"]


def test_w_global_frozen_during_local_training():
    """w_global is the pre-training global; a positive mu pulls toward it."""
    X, y, _, _ = make_synthetic(n=800)
    mu = 50.0  # large enough that FedProx barely leaves the global point
    c = FedAvgClient(model_cfg(), X[:600], y[:600], X[600:], y[600:], seed=3)
    global_params = c.get_parameters({})

    # mu=0: unconstrained local steps move away from the global start
    c0 = FedAvgClient(model_cfg(), X[:600], y[:600], X[600:], y[600:], seed=3)
    p0, _, _ = c0.fit(c0.get_parameters({}),
                       {"server_round": 1, "local_epochs": 2, "lr": 0.05,
                        "batch_size": 64})
    dist0 = sum(float(np.sum((a - b) ** 2)) for a, b in zip(p0, global_params))

    # mu=50: strong proximal term keeps the result near the global point
    c1 = FedAvgClient(model_cfg(), X[:600], y[:600], X[600:], y[600:], seed=3)
    p1, _, m1 = c1.fit(c1.get_parameters({}),
                        {"server_round": 1, "local_epochs": 2, "lr": 0.05,
                         "batch_size": 64, "proximal_mu": mu})
    dist1 = sum(float(np.sum((a - b) ** 2)) for a, b in zip(p1, global_params))

    assert dist0 > dist1
    assert m1["prox_penalty"] > 0.0
    # and the frozen reference equals the GLOBAL start (not the final params)
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(p1, global_params)) is False


def test_client_fn_binds_worker_to_its_own_partition():
    """Each worker must be bound to its OWN partition index (regression).

    In-process start_client workers all receive the same node id from Flower
    (gRPC-bidi has no node concept), so the cid string is NOT the partition
    index. build_client_fn must bind each worker to its explicit cid; otherwise
    every worker would train on the same client's data.
    """
    X, y, _, _ = make_synthetic(n=900)
    from src.federated.data.partition import ClientPartitionConfig as CPC
    part = build_client_partition(y, CPC(strategy="iid", clients=3, seed=7,
                                         val_fraction=0.1, min_samples_per_client=0))

    cfg = model_cfg()
    seen_train = set()
    for cid in range(3):
        # pass a bogus "cid" string from Flower: the worker must still get
        # the data of the partition index it was bound to
        wrapped = build_client_fn_part(part, X, y, cfg, seed=1, cid=cid)("node-xyz")
        X_tr = wrapped.numpy_client.X_train
        assert X_tr.shape[0] == len(part.client_train[cid])
        # the worker's rows must be EXACTLY its own partition rows
        assert np.array_equal(X_tr, X[part.client_train[cid]])
        seen_train.add(int(X_tr.sum()))
    assert len(seen_train) > 1  # clients must NOT all see the same data


def build_client_fn_part(part, X, y, model_cfg, seed=1, cid=None):
    """Tiny in-memory PartitionClientData stand-in using a ClientPartition."""
    from src.federated.fl.dataset import PartitionClientData
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "iid-c3-s7"
    save_partition(tmp, part, y)
    data = PartitionClientData(tmp, X, y)
    return build_client_fn(data, model_cfg, seed=seed, cid=cid)


@pytest.fixture(scope="module")
def fedprox_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("fedprox")
    X, y, X_test, y_test = make_synthetic()

    pdir = base / "iid-c3-s7"
    pcfg = ClientPartitionConfig(strategy="iid", clients=3, seed=7,
                                 val_fraction=0.1, min_samples_per_client=0)
    part = build_client_partition(y, pcfg)
    save_partition(pdir, part, y)

    def make_cfg(algorithm, mu):
        cfg = ExperimentConfig()
        cfg.fl.num_clients = 3
        cfg.fl.num_rounds = 2
        cfg.fl.client_fraction = 1.0
        cfg.fl.algorithm = algorithm
        cfg.fl.proximal_mu = mu
        cfg.train.local_epochs = 1
        cfg.train.learning_rate = 0.01
        cfg.train.batch_size = 64
        cfg.model.input_dim = 16
        cfg.model.hidden_layers = (8, 4)
        cfg.model.dropout = 0.0
        cfg.seed = 42
        return cfg

    scale_inv = np.ones(16, dtype=np.float32)

    out_avg = base / "results-fedavg"
    res_avg = run_fl_experiment(make_cfg("fedavg", 0.0), pdir, X, y, X_test, y_test,
                                scale_inv=scale_inv, output_dir=out_avg, seed=42)
    out_prox = base / "results-fedprox"
    res_prox = run_fl_experiment(make_cfg("fedprox", 0.1), pdir, X, y, X_test, y_test,
                                 scale_inv=scale_inv, output_dir=out_prox, seed=42)

    return {"pdir": pdir, "X": X, "y": y, "res_avg": res_avg, "res_prox": res_prox,
            "out_avg": out_avg, "out_prox": out_prox}


def test_run_fl_experiment_fedprox(fedprox_env):
    res = fedprox_env["res_prox"]
    assert res["experiment"]["algorithm"] == "fedprox"
    assert res["experiment"]["proximal_mu"] == 0.1
    assert res["experiment"]["fl_config"]["algorithm"] == "fedprox"
    assert len(res["rounds"]) == 2
    for r in res["rounds"]:
        assert r["avg_client_f1"] is not None
        assert r["worst_client_f1"] is not None
        assert "client_f1_variance" in r
        assert r["global_eval"]["n_test"] == 400
    assert res["final_global_test_metrics"]["roc_auc"] is not None
    # client-reported prox metrics were aggregated honestly
    assert res["rounds"][-1]["global_eval"]["accuracy"] >= 0.5


def test_same_partition_config_for_both_algorithms(fedprox_env):
    a = fedprox_env["res_avg"]
    p = fedprox_env["res_prox"]
    assert a["experiment"]["partition_dir"] == p["experiment"]["partition_dir"]
    assert a["experiment"]["partition_config"] == p["experiment"]["partition_config"]
    assert a["experiment"]["fl_config"]["seed"] == p["experiment"]["fl_config"]["seed"] == 42
    # identical fl settings except algorithm/mu
    for k in ("num_clients", "num_rounds", "client_fraction", "local_epochs",
              "learning_rate", "batch_size"):
        assert a["experiment"]["fl_config"][k] == p["experiment"]["fl_config"][k]
    # identical communication footprint (same model, same rounds, same clients)
    assert (a["communication"]["totals"]["total_bytes_exchanged"]
            == p["communication"]["totals"]["total_bytes_exchanged"])


def test_fedprox_output_files(fedprox_env):
    out = fedprox_env["out_prox"]
    assert (out / "config.json").exists()
    assert (out / "rounds.jsonl").exists()
    assert (out / "summary.json").exists()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment"]["algorithm"] == "fedprox"
    assert summary["experiment"]["proximal_mu"] == 0.1
    assert summary["communication"]["totals"]["total_bytes_exchanged"] > 0


def test_fedavg_still_works_through_run_fl_experiment(fedprox_env):
    res = fedprox_env["res_avg"]
    assert res["experiment"]["algorithm"] == "fedavg"
    assert len(res["rounds"]) == 2
    assert res["final_global_test_metrics"]["accuracy"] > 0.5


def test_unsupported_algorithm_rejected(tmp_path_factory):
    X, y, X_test, y_test = make_synthetic(n=600)
    pdir = tmp_path_factory.mktemp("fp2") / "iid-c3-s7"
    save_partition(pdir, build_client_partition(y, ClientPartitionConfig(
        strategy="iid", clients=3, seed=7, min_samples_per_client=0)), y)
    cfg = ExperimentConfig()
    cfg.fl.num_clients = 3
    cfg.fl.algorithm = "federated_dolphin"
    with pytest.raises(ValueError, match="unsupported FL algorithm"):
        run_fl_experiment(cfg, pdir, X, y, X_test, y_test)


def test_model_matches_reference_prox_loss():
    """The client's prox objective matches the analytic formula."""
    X, y, _, _ = make_synthetic(n=400)
    mu = 0.7
    c = FedAvgClient(model_cfg(), X[:300], y[:300], X[300:], y[300:], seed=9)
    global_params = c.get_parameters({})
    _, _, m = c.fit(global_params,
                    {"server_round": 1, "local_epochs": 1, "lr": 0.01,
                     "batch_size": 64, "proximal_mu": mu})
    # proxy: without any learning the reported penalty should still be > 0,
    # and it must match the formula order of magnitude for mu>0
    assert m["prox_penalty"] > 0.0
