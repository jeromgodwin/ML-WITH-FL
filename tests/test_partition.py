"""Phase 4: client partition tests — reproducibility, isolation, coverage,
distribution differences (IID vs skew strategies)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.federated.data.partition import (  # noqa: E402
    STRATEGIES, ClientPartitionConfig, build_client_partition, load_partition,
    partition_report, save_partition, verify_partition,
)


def _y(n=4000, mal_frac=0.5, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    y = np.zeros(n, dtype=np.int8)
    y[: int(n * mal_frac)] = 1
    return y[rng.permutation(n)]


def _families_for(y, n_fams=8):
    """Aligned family labels: only malicious rows get a family (as in EMBER)."""
    fams = np.array([""] * len(y), dtype=object)
    mal = np.flatnonzero(y == 1)
    per = len(mal) // n_fams
    for i in range(n_fams):
        fams[mal[i * per:(i + 1) * per]] = f"f{i}"
    return fams


def _build(y, **cfg):
    cfg.setdefault("min_samples_per_client", 0)
    fams = cfg.pop("families", None)
    return build_client_partition(y, ClientPartitionConfig(**cfg), family_labels=fams)


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------

def _build2(y, kw):
    fams = kw.pop("families", None)
    return build_client_partition(y, ClientPartitionConfig(**kw), family_labels=fams)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_reproducible_same_seed(strategy):
    y = _y()
    kw = {"strategy": strategy, "clients": 8, "seed": 11, "min_samples_per_client": 150}
    if strategy == "family_skew":
        kw["families"] = _families_for(y)
    p1 = _build2(y, dict(kw))
    p2 = _build2(y, dict(kw))
    for c in range(8):
        assert np.array_equal(p1.client_train[c], p2.client_train[c])
        assert np.array_equal(p1.client_val[c], p2.client_val[c])
    assert np.array_equal(p1.pool_idx, p2.pool_idx)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_different_seed_differs(strategy):
    y = _y()
    kw = {"strategy": strategy, "clients": 8, "seed": 11, "min_samples_per_client": 150}
    if strategy == "family_skew":
        kw["families"] = _families_for(y)
    p1 = _build2(y, dict(kw))
    p2 = _build2(y, dict({**kw, "seed": 12}))
    assert any(not np.array_equal(p1.client_train[c], p2.client_train[c]) for c in range(8))


# ---------------------------------------------------------------------------
# isolation and coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", STRATEGIES)
def test_strict_partition_of_labeled_pool(strategy):
    y = _y()
    y[::9] = -1  # sprinkle unlabeled rows: never in any client
    kw = {"strategy": strategy, "clients": 8, "seed": 3, "min_samples_per_client": 150}
    if strategy == "family_skew":
        kw["families"] = _families_for(y)
    p = _build2(y, kw)

    labeled = set(np.flatnonzero((y == 0) | (y == 1)).tolist())
    assert set(p.pool_idx.tolist()) == labeled  # complete labeled coverage
    union = set()
    for c in range(8):
        tr, va = p.client_train[c], p.client_val[c]
        assert set(tr).isdisjoint(set(va))  # train/val disjoint per client
        union |= set(tr.tolist()) | set(va.tolist())
        for d in range(c + 1, 8):
            assert set(tr.tolist()).isdisjoint(set(p.client_train[d].tolist()))
            assert set(tr.tolist()).isdisjoint(set(p.client_val[d].tolist()))
    assert union == labeled  # every labeled row covered exactly once
    checks = verify_partition(p, y)
    assert checks["verified"]
    assert checks["cross_client_overlap"] == 0
    assert checks["train_val_overlap"] == 0


def test_pool_cap_respected():
    y = _y()
    p = _build(y, strategy="iid", clients=8, max_samples=1200, seed=5)
    assert len(p.pool_idx) == 1200
    assert verify_partition(p, y)["verified"]


def test_pool_idx_restriction():
    y = _y()
    y[::9] = -1
    train_only = np.flatnonzero(y != -1)[::2]  # arbitrary labeled subset
    p = build_client_partition(y, ClientPartitionConfig(strategy="iid", clients=8, seed=5,
                                                         min_samples_per_client=0),
                               pool_idx=train_only)
    assert set(p.pool_idx.tolist()) == set(train_only.tolist())
    assert verify_partition(p, y)["verified"]
    with pytest.raises(ValueError):
        build_client_partition(y, ClientPartitionConfig(strategy="iid", clients=8, seed=5,
                                                         min_samples_per_client=0),
                               pool_idx=np.array([0]))  # row 0 is unlabeled


def test_min_samples_enforced():
    y = _y()
    p = _build(y, strategy="quantity_skew", clients=8, seed=7, min_samples_per_client=150)
    sizes = p.client_sizes()
    assert (sizes >= 150).all()
    assert sizes.sum() == len(p.pool_idx)


def test_min_samples_impossible_raises():
    y = _y(400)
    with pytest.raises(ValueError):
        _build(y, strategy="iid", clients=8, seed=7, min_samples_per_client=100)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_val_fraction_respected(strategy):
    y = _y()
    kw = {"strategy": strategy, "clients": 8, "seed": 9, "val_fraction": 0.2,
          "min_samples_per_client": 200}
    if strategy == "family_skew":
        kw["families"] = _families_for(y)
    p = _build2(y, kw)
    for c in range(8):
        n = len(p.client_train[c]) + len(p.client_val[c])
        if n:
            frac = len(p.client_val[c]) / n
            assert 0.1 <= frac <= 0.3, f"client {c} val fraction {frac:.3f}"


def test_family_skew_requires_labels():
    with pytest.raises(ValueError):
        _build(_y(), strategy="family_skew", clients=8, seed=1)


# ---------------------------------------------------------------------------
# distribution differences
# ---------------------------------------------------------------------------

def _kl_mean(p):
    return p["divergence"]["class_kl_mean_weighted"]


def test_iid_is_balanced_and_label_iid():
    y = _y()
    p = _build(y, strategy="iid", clients=8, seed=4)
    report = partition_report(p, y)
    for cl in report["clients"]:
        assert abs(cl["malware_frac"] - 0.5) < 0.1
    assert _kl_mean(report) < 0.05
    assert report["divergence"]["count_cv"] < 0.1


def test_skew_is_monotonic_mild_moderate_severe():
    y = _y(6000)
    kl = {}
    for s in ("iid", "mild", "moderate", "severe"):
        p = _build(y, strategy=s, clients=10, seed=42, min_samples_per_client=100)
        kl[s] = _kl_mean(partition_report(p, y))
    assert kl["iid"] < kl["mild"] < kl["moderate"] < kl["severe"]


def test_label_skew_creates_class_imbalance():
    y = _y(6000)
    p = _build(y, strategy="label_skew", clients=10, seed=42, min_samples_per_client=100)
    report = partition_report(p, y)
    assert _kl_mean(report) > _kl_mean(partition_report(
        _build(y, strategy="iid", clients=10, seed=42), y))
    assert max(abs(cl["malware_frac"] - 0.5) for cl in report["clients"]) > 0.3


def test_quantity_skew_increases_count_imbalance():
    y = _y()
    p_iid = _build(y, strategy="iid", clients=8, seed=21)
    p_q = _build(y, strategy="quantity_skew", clients=8, seed=21)
    cv_iid = partition_report(p_iid, y)["divergence"]["count_cv"]
    cv_q = partition_report(p_q, y)["divergence"]["count_cv"]
    assert cv_q > cv_iid * 2
    assert cv_q > 0.5


def test_family_skew_concentrates_families():
    y = _y()
    fams = _families_for(y)
    p_iid = _build(y, strategy="iid", clients=8, seed=2)
    p_f = _build(y, strategy="family_skew", clients=8, seed=2, families=fams)
    ent_iid = partition_report(p_iid, y, fams)["divergence"]["family_entropy_mean"]
    ent_f = partition_report(p_f, y, fams)["divergence"]["family_entropy_mean"]
    assert ent_f < ent_iid

    # each family must sit on a single client
    for c in range(8):
        idx = np.concatenate([p_f.client_train[c], p_f.client_val[c]])
        my_fams = set(fams[idx]) - {""}
        for d in range(8):
            if d == c:
                continue
            other = np.concatenate([p_f.client_train[d], p_f.client_val[d]])
            assert my_fams.isdisjoint(set(fams[other]) - {""})


def test_combined_severe_has_both_skews():
    y = _y(6000)
    p = _build(y, strategy="combined_severe", clients=10, seed=42, min_samples_per_client=100)
    report = partition_report(p, y)
    d = report["divergence"]
    p_iid = partition_report(_build(y, strategy="iid", clients=10, seed=42), y)
    assert d["class_kl_mean_weighted"] > 5 * p_iid["divergence"]["class_kl_mean_weighted"]
    assert d["count_cv"] > p_iid["divergence"]["count_cv"]
    assert max(abs(cl["malware_frac"] - 0.5) for cl in report["clients"]) > 0.3


# ---------------------------------------------------------------------------
# persistence round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["iid", "severe", "family_skew"])
def test_save_load_roundtrip(tmp_path, strategy):
    y = _y()
    fams = _families_for(y)
    p = build_client_partition(
        y, ClientPartitionConfig(strategy=strategy, clients=8, seed=6,
                                 min_samples_per_client=150),
        family_labels=fams if strategy == "family_skew" else None)
    out = tmp_path / "partition"
    report = save_partition(out, p, y, fams if strategy == "family_skew" else None)

    assert (out / "client_indices.npz").exists()
    assert (out / "manifest.json").exists()
    assert (out / "report.json").exists()
    assert report["isolation"]["verified"]

    cfg, pool, train, val, fam = load_partition(out)
    assert cfg.strategy == strategy
    assert np.array_equal(pool, p.pool_idx)
    for c in range(8):
        assert np.array_equal(train[c], p.client_train[c])
        assert np.array_equal(val[c], p.client_val[c])
    if strategy == "family_skew":
        assert fam is not None and len(fam) == len(pool)


def test_report_shape():
    y = _y()
    p = _build(y, strategy="severe", clients=8, seed=4, min_samples_per_client=150)
    report = partition_report(p, y, _families_for(y))
    assert set(report) >= {"strategy", "config", "pool", "divergence", "clients"}
    cl = report["clients"][0]
    assert set(cl) >= {"client", "n_samples", "n_benign", "n_malicious",
                       "benign_frac", "malware_frac", "family_distribution",
                       "family_entropy"}
    assert sum(c["n_samples"] for c in report["clients"]) == report["pool"]["n"]