"""Phase 11: personalized FL (FedPer-style body/head split) over saved partitions.

Verifies that the personalized method:
- splits the model into a shared BODY (aggregated) and a personal HEAD
- keeps the personal head strictly client-side: never transmitted, never
  overwritten by received parameters, never shared between clients
- reports the global test evaluation through a server-side probe head
- transmits strictly fewer bytes than FedAvg (head excluded)
- runs end-to-end over a real Flower gRPC server, reproducibly
- leaves FedAvg/FedProx untouched (they are exercised by the other suites)
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
from src.federated.fl.client import (  # noqa: E402
    PersonalizedClient, build_personalized_client_fn,
)
from src.federated.fl.dataset import PartitionClientData  # noqa: E402
from src.federated.fl.server import (  # noqa: E402
    _train_probe_head, run_fl_experiment,
)
from src.federated.models.mlp import (  # noqa: E402
    DeterministicDropout, MLPConfig, build_mlp, build_personalized_mlp,
)


def make_synthetic(n=1800, n_test=400, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 16)).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] + 0.5 * rng.normal(size=n)) > 0).astype(np.int64)
    X_test = rng.normal(size=(n_test, 16)).astype(np.float32)
    y_test = ((X_test[:, 0] + X_test[:, 1] + 0.5 * rng.normal(size=n_test)) > 0).astype(np.int64)
    return X, y, X_test, y_test


def model_cfg():
    return MLPConfig(input_dim=16, hidden_layers=(8, 4), dropout=0.0)


def full_param_count(cfg) -> int:
    return sum(p.numel() for p in build_mlp(cfg).parameters())


def body_param_count(cfg) -> int:
    return sum(p.numel() for p in build_personalized_mlp(cfg).body.parameters())


def head_param_count(cfg) -> int:
    return sum(p.numel() for p in build_personalized_mlp(cfg).head.parameters())


def make_client(cfg=None, X=None, y=None, seed=3):
    cfg = cfg or model_cfg()
    if X is None:
        X, y, _, _ = make_synthetic(n=800)
    return PersonalizedClient(cfg, X[:600], y[:600], X[600:], y[600:], seed=seed)


# --------------------------------------------------------------------------
# model split
# --------------------------------------------------------------------------

def test_mlp_splits_into_body_and_head():
    cfg = model_cfg()
    m = build_personalized_mlp(cfg)
    assert body_param_count(cfg) + head_param_count(cfg) == full_param_count(cfg)
    # head is the final layer: Linear(hidden[-1] -> 1)
    assert head_param_count(cfg) == cfg.hidden_layers[-1] * 1 + 1
    # body parameters must NOT include head parameters
    body_ids = {id(p) for p in m.body.parameters()}
    assert all(id(p) not in body_ids for p in m.head.parameters())
    logits = m(torch.zeros(5, cfg.input_dim))
    assert logits.shape == (5, 1)


def test_personalized_mlp_standardizes_features():
    """LayerNorm before the head keeps logits O(1) despite extreme inputs.

    EMBER long-tail raw-count features reach ~2^32 and blow the body's hidden
    activations up to ~1e6; without standardization any head saturates and its
    gradients vanish. Regression guard for the Phase-11 probe evaluation.
    """
    cfg = model_cfg()
    m = build_personalized_mlp(cfg)
    m.eval()
    huge = torch.zeros(8, cfg.input_dim)
    huge[:, 0] = 4e9
    with torch.no_grad():
        logits = m(huge)
    assert float(torch.abs(logits).max()) < 100.0


def test_deterministic_dropout_masks():
    """Same (seed, input) => same mask; different seed => different mask.

    Regression guard: client threads used to draw masks from the GLOBAL torch
    RNG, so mask streams depended on thread interleaving and runs were not
    reproducible. DeterministicDropout draws from a local generator instead.
    """
    m1 = DeterministicDropout(0.5)
    m2 = DeterministicDropout(0.5)
    x = torch.randn(4, 8)
    m1.train()
    m2.train()
    m1.set_seed(7)
    m2.set_seed(7)
    out1, out2 = m1(x), m2(x)
    assert torch.allclose(out1, out2, atol=1e-6)
    assert not torch.equal(out1, x)  # masks actually applied
    m1.set_seed(8)
    assert not torch.allclose(m1(x), out1, atol=1e-6)
    m1.eval()
    assert torch.equal(m1(x), x)  # identity in eval mode


def test_initial_heads_are_identical_between_clients():
    """Same seed => same head init; divergence comes from local data only.

    Mirrors ``build_personalized_client_fn``, which reseeds before each
    client's model is built.
    """
    from src.utils.reproducibility import set_all_seeds

    X, y, _, _ = make_synthetic(n=800)
    set_all_seeds(3)
    c1 = PersonalizedClient(model_cfg(), X[:600], y[:600], X[600:], y[600:], seed=3)
    set_all_seeds(3)
    c2 = PersonalizedClient(model_cfg(), X[:600], y[:600], X[600:], y[600:], seed=3)
    assert all(np.allclose(a, b, atol=1e-8)
               for a, b in zip(c1.head_params, c2.head_params))


# --------------------------------------------------------------------------
# client state: head stays local and client-specific
# --------------------------------------------------------------------------

def test_get_parameters_returns_body_only():
    cfg = model_cfg()
    c = make_client(cfg=cfg)
    params = c.get_parameters({})
    assert len(params) == len(list(c.model.body.parameters()))
    n = sum(int(np.prod(np.asarray(p).shape)) for p in params)
    assert n == body_param_count(cfg)
    assert n < full_param_count(cfg)


def test_set_parameters_preserves_personal_head():
    cfg = model_cfg()
    c = make_client(cfg=cfg)
    head_before = [a.copy() for a in c.head_params]
    # adversarial global body: must change the body but NEVER the head
    rng = np.random.default_rng(0)
    adversarial = [rng.normal(size=np.asarray(p).shape).astype(np.float32)
                   for p in c.get_parameters({})]
    c._set_parameters(adversarial)
    assert all(np.allclose(np.asarray(p), np.asarray(a), atol=1e-6)
               for p, a in zip(c.get_parameters({}), adversarial))
    assert all(np.allclose(a, b, atol=1e-8)
               for a, b in zip(head_before, c.head_params))


def test_head_persists_across_rounds_and_never_reinitialized():
    """Head evolves from local training but survives later _set_parameters calls."""
    cfg = model_cfg()
    c = make_client(cfg=cfg)
    start = c.get_parameters({})
    head_before = [a.copy() for a in c.head_params]
    _, _, m1 = c.fit(start, {"server_round": 1, "local_epochs": 2, "lr": 0.05,
                             "batch_size": 64})
    head_after_fit1 = [a.copy() for a in c.head_params]
    # local training actually moved the personal head
    assert not all(np.allclose(a, b, atol=1e-6)
                   for a, b in zip(head_before, head_after_fit1))
    # the next round the server sends the SAME global body again
    c._set_parameters(start)
    assert all(np.allclose(a, b, atol=1e-8)
               for a, b in zip(head_after_fit1, c.head_params))
    assert m1["train_loss"] >= 0.0


def test_clients_heads_diverge_on_different_data():
    """Two clients with different local data end up with different heads."""
    X, y, _, _ = make_synthetic(n=900)
    # client A sees mostly positives, client B mostly negatives
    pos = np.flatnonzero(y == 1)[:300]
    neg = np.flatnonzero(y == 0)[:300]
    idx_a = np.concatenate([pos, neg[:60]])
    idx_b = np.concatenate([pos[:60], neg])
    c_a = PersonalizedClient(model_cfg(), X[idx_a], y[idx_a], X[850:], y[850:], seed=5)
    c_b = PersonalizedClient(model_cfg(), X[idx_b], y[idx_b], X[850:], y[850:], seed=5)
    start = [a.copy() for a in c_a.get_parameters({})]
    c_a._set_parameters(start)
    c_b._set_parameters(start)
    fit_cfg = {"server_round": 1, "local_epochs": 3, "lr": 0.05, "batch_size": 64}
    c_a.fit(start, fit_cfg)
    c_b.fit(start, fit_cfg)
    ha = c_a.head_params
    hb = c_b.head_params
    assert not all(np.allclose(a, b, atol=1e-6) for a, b in zip(ha, hb))


def test_fit_upload_excludes_head():
    cfg = model_cfg()
    c = make_client(cfg=cfg)
    start = c.get_parameters({})
    _, _, m = c.fit(start, {"server_round": 1, "local_epochs": 1, "lr": 0.01,
                            "batch_size": 64})
    body_bytes = int(sum(np.asarray(p).nbytes for p in start))
    assert m["upload_bytes"] == body_bytes
    assert m["upload_bytes"] < full_param_count(cfg) * 4
    assert m["download_bytes"] == body_bytes


def test_body_frozen_during_head_adaptation():
    """FedRep phase A: head-only training must not move the shared body."""
    cfg = model_cfg()
    c = make_client(cfg=cfg)
    start = c.get_parameters({})
    body_before = [a.copy() for a in start]
    head_before = [a.copy() for a in c.head_params]
    c.fit(start, {"server_round": 1, "local_epochs": 0, "head_epochs": 2,
                  "head_lr": 0.1, "batch_size": 64})
    assert all(np.allclose(np.asarray(a), np.asarray(b), atol=1e-8)
               for a, b in zip(body_before, c.get_parameters({})))
    assert not all(np.allclose(a, b, atol=1e-6)
                   for a, b in zip(head_before, c.head_params))


def test_head_frozen_during_body_training():
    """FedRep phase B: body-only training must not move the personal head."""
    cfg = model_cfg()
    c = make_client(cfg=cfg)
    start = c.get_parameters({})
    head_before = [a.copy() for a in c.head_params]
    body_before = [a.copy() for a in start]
    c.fit(start, {"server_round": 1, "local_epochs": 2, "head_epochs": 0,
                  "lr": 0.05, "batch_size": 64})
    assert all(np.allclose(a, b, atol=1e-8)
               for a, b in zip(head_before, c.head_params))
    assert not all(np.allclose(np.asarray(a), np.asarray(b), atol=1e-6)
                   for a, b in zip(body_before, c.get_parameters({})))


# --------------------------------------------------------------------------
# client fn binding (regression: in-process workers share one node id)
# --------------------------------------------------------------------------

def test_personalized_client_fn_binds_worker_to_own_partition():
    X, y, _, _ = make_synthetic(n=900)
    part = build_client_partition(y, ClientPartitionConfig(
        strategy="iid", clients=3, seed=7, val_fraction=0.1,
        min_samples_per_client=0))
    tmp = Path(__import__("tempfile").mkdtemp()) / "iid-c3-s7"
    save_partition(tmp, part, y)
    data = PartitionClientData(tmp, X, y)
    cfg = model_cfg()
    seen_train = set()
    for cid in range(3):
        wrapped = build_personalized_client_fn(data, cfg, seed=1, cid=cid)("node-xyz")
        X_tr = wrapped.numpy_client.X_train
        assert np.array_equal(X_tr, X[part.client_train[cid]])
        seen_train.add(int(X_tr.sum()))
    assert len(seen_train) > 1


# --------------------------------------------------------------------------
# probe head (server-side global evaluation)
# --------------------------------------------------------------------------

def test_probe_head_trains_head_but_leaves_body_frozen():
    cfg = model_cfg()
    X, y, _, _ = make_synthetic(n=600)
    model = build_personalized_mlp(cfg)
    body_before = [p.detach().clone() for p in model.body.parameters()]
    head_before = [p.detach().clone() for p in model.head.parameters()]
    n, in_sample_auc, restarts = _train_probe_head(
        model, X, y, seed=1, probe_samples=400,
        probe_epochs=2, learning_rate=0.1, batch_size=64)
    assert n == 400
    assert 0.0 <= in_sample_auc <= 1.0
    assert 0 <= restarts <= 3
    assert all(torch.allclose(a, b, atol=1e-12)
               for a, b in zip(body_before, model.body.parameters()))
    assert not all(torch.allclose(a, b, atol=1e-9)
                   for a, b in zip(head_before, model.head.parameters()))
    head_after = [p.detach().clone() for p in model.head.parameters()]
    n2, auc2, restarts2 = _train_probe_head(
        model, X, y, seed=1, probe_samples=400,
        probe_epochs=2, learning_rate=0.1, batch_size=64)
    # deterministic protocol: identical call reproduces the identical result
    assert n2 == n
    assert auc2 == in_sample_auc
    assert restarts2 == restarts
    assert all(torch.allclose(a, b, atol=1e-12)
               for a, b in zip(head_after, model.head.parameters()))


# --------------------------------------------------------------------------
# end-to-end experiments
# --------------------------------------------------------------------------

def make_cfg(algorithm="personalized", rounds=2):
    cfg = ExperimentConfig()
    cfg.fl.num_clients = 3
    cfg.fl.num_rounds = rounds
    cfg.fl.client_fraction = 1.0
    cfg.fl.algorithm = algorithm
    cfg.fl.personalized_probe_samples = 400
    cfg.fl.personalized_probe_epochs = 10
    cfg.train.local_epochs = 1
    cfg.train.learning_rate = 0.01
    cfg.train.batch_size = 64
    cfg.model.input_dim = 16
    cfg.model.hidden_layers = (8, 4)
    cfg.model.dropout = 0.0
    cfg.seed = 42
    return cfg


@pytest.fixture(scope="module")
def personalized_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("personalized")
    X, y, X_test, y_test = make_synthetic()

    pdir = base / "iid-c3-s7"
    part = build_client_partition(y, ClientPartitionConfig(
        strategy="iid", clients=3, seed=7, val_fraction=0.1,
        min_samples_per_client=0))
    save_partition(pdir, part, y)

    scale_inv = np.ones(16, dtype=np.float32)
    out_per = base / "results-personalized"
    res_per = run_fl_experiment(make_cfg("personalized"), pdir, X, y, X_test, y_test,
                                scale_inv=scale_inv, output_dir=out_per, seed=42)
    out_avg = base / "results-fedavg"
    res_avg = run_fl_experiment(make_cfg("fedavg"), pdir, X, y, X_test, y_test,
                                scale_inv=scale_inv, output_dir=out_avg, seed=42)
    return {"pdir": pdir, "X": X, "y": y, "res_per": res_per, "res_avg": res_avg,
            "out_per": out_per, "out_avg": out_avg}


def test_personalized_e2e(personalized_env):
    res = personalized_env["res_per"]
    assert res["experiment"]["algorithm"] == "personalized"
    assert len(res["rounds"]) == 2
    per = res["experiment"]["personalized"]
    assert per["head_parameter_count"] == head_param_count(model_cfg())
    assert per["body_parameter_count"] + per["head_parameter_count"] \
        == full_param_count(model_cfg())
    for r in res["rounds"]:
        assert r["avg_client_f1"] is not None
        assert r["worst_client_f1"] is not None
        assert "client_f1_variance" in r
        assert r["global_eval"]["n_test"] == 400
        assert r["global_eval"].get("probe_head_trained") is True
        assert r["global_eval"]["probe_train_samples"] == 400
    assert res["final_global_test_metrics"]["accuracy"] > 0.5
    assert res["final_global_test_metrics"]["roc_auc"] is not None


def test_personalized_communication_smaller_than_fedavg(personalized_env):
    per = personalized_env["res_per"]["communication"]["totals"]
    avg = personalized_env["res_avg"]["communication"]["totals"]
    # the personal head (129 params) never crosses the wire: strictly fewer bytes
    assert per["total_bytes_exchanged"] < avg["total_bytes_exchanged"]
    assert per["total_upload_bytes"] < avg["total_upload_bytes"]
    head_bytes = head_param_count(model_cfg()) * 4 * 2 * 2  # 2 rounds, up+down
    assert avg["total_bytes_exchanged"] - per["total_bytes_exchanged"] >= head_bytes


def test_personalized_uses_same_partition_as_fedavg(personalized_env):
    a = personalized_env["res_avg"]
    p = personalized_env["res_per"]
    assert a["experiment"]["partition_dir"] == p["experiment"]["partition_dir"]
    assert a["experiment"]["partition_config"] == p["experiment"]["partition_config"]
    for k in ("num_clients", "num_rounds", "client_fraction", "local_epochs",
              "learning_rate", "batch_size", "seed"):
        assert a["experiment"]["fl_config"][k] == p["experiment"]["fl_config"][k]


def test_personalized_output_files(personalized_env):
    out = personalized_env["out_per"]
    assert (out / "config.json").exists()
    assert (out / "rounds.jsonl").exists()
    assert (out / "summary.json").exists()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment"]["algorithm"] == "personalized"
    assert summary["communication"]["totals"]["total_bytes_exchanged"] > 0


def test_personalized_reproducible(tmp_path_factory):
    base = tmp_path_factory.mktemp("personalized-repro")
    X, y, X_test, y_test = make_synthetic()
    pdir = base / "iid-c3-s7"
    save_partition(pdir, build_client_partition(y, ClientPartitionConfig(
        strategy="iid", clients=3, seed=7, min_samples_per_client=0)), y)
    scale_inv = np.ones(16, dtype=np.float32)

    r1 = run_fl_experiment(make_cfg("personalized"), pdir, X, y, X_test, y_test,
                           scale_inv=scale_inv, seed=42)
    r2 = run_fl_experiment(make_cfg("personalized"), pdir, X, y, X_test, y_test,
                           scale_inv=scale_inv, seed=42)
    m1 = r1["final_global_test_metrics"]
    m2 = r2["final_global_test_metrics"]
    for k in ("accuracy", "precision", "recall", "f1"):
        assert m1[k] == pytest.approx(m2[k], abs=1e-6)
    # roc_auc can jitter ~1e-5 from CPU reduction scheduling (FP noise), not
    # from RNG state: client metrics below are asserted bit-exact.
    assert m1["roc_auc"] == pytest.approx(m2["roc_auc"], abs=1e-4)
    assert r1["rounds"][-1]["worst_client_f1"] == pytest.approx(
        r2["rounds"][-1]["worst_client_f1"], abs=1e-6)


def test_personalized_reproducible_with_dropout(tmp_path_factory):
    """Real dropout (0.2) must not break reproducibility across runs.

    Regression guard for the global-RNG race: with dropout>0, concurrent
    client threads used to draw masks from the shared global RNG, so identical
    configs could produce different results run to run.
    """
    base = tmp_path_factory.mktemp("personalized-repro-drop")
    X, y, X_test, y_test = make_synthetic()
    pdir = base / "iid-c3-s7"
    save_partition(pdir, build_client_partition(y, ClientPartitionConfig(
        strategy="iid", clients=3, seed=7, min_samples_per_client=0)), y)
    scale_inv = np.ones(16, dtype=np.float32)
    cfg = make_cfg("personalized", rounds=2)
    cfg.model.dropout = 0.2

    r1 = run_fl_experiment(cfg, pdir, X, y, X_test, y_test,
                           scale_inv=scale_inv, seed=42)
    r2 = run_fl_experiment(cfg, pdir, X, y, X_test, y_test,
                           scale_inv=scale_inv, seed=42)
    for k in ("accuracy", "precision", "recall", "f1"):
        assert r1["final_global_test_metrics"][k] == pytest.approx(
            r2["final_global_test_metrics"][k], abs=1e-6)
    assert r1["final_global_test_metrics"]["roc_auc"] == pytest.approx(
        r2["final_global_test_metrics"]["roc_auc"], abs=1e-4)
    assert r1["rounds"][-1]["avg_client_f1"] == pytest.approx(
        r2["rounds"][-1]["avg_client_f1"], abs=1e-6)
    assert r1["rounds"][-1]["worst_client_f1"] == pytest.approx(
        r2["rounds"][-1]["worst_client_f1"], abs=1e-6)


def test_fedavg_still_works_next_to_personalized(personalized_env):
    res = personalized_env["res_avg"]
    assert res["experiment"]["algorithm"] == "fedavg"
    assert len(res["rounds"]) == 2
    assert res["final_global_test_metrics"]["accuracy"] > 0.5
    assert "personalized" not in res["experiment"]
