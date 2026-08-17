"""Split tests: seeded reproducible train/val indices, labels respected."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.federated.data.split import (  # noqa: E402
    check_split_reproducible, load_split_indices, make_train_val_indices,
)


def _y(n=100, mal_frac=0.5):
    y = np.zeros(n, dtype=np.int8)
    y[: int(n * mal_frac)] = 1
    y[::7] = -1  # sprinkle unlabeled rows
    rng = np.random.default_rng(0)
    return y[rng.permutation(n)]


def test_split_seeded_reproducible(tmp_path):
    y = _y()
    p1 = tmp_path / "a.npz"
    p2 = tmp_path / "b.npz"
    r1 = make_train_val_indices(y, p1, seed=7, val_fraction=0.2)
    r2 = make_train_val_indices(y, p2, seed=7, val_fraction=0.2)
    assert np.array_equal(r1["train_idx"], r2["train_idx"])
    assert np.array_equal(r1["val_idx"], r2["val_idx"])
    assert check_split_reproducible(p2, seed=7, val_fraction=0.2)
    assert not check_split_reproducible(p2, seed=8, val_fraction=0.2)


def test_split_val_is_labeled_and_clean(tmp_path):
    y = _y()
    out = tmp_path / "s.npz"
    result = make_train_val_indices(y, out, seed=3, val_fraction=0.2)
    assert len(result["val_idx"]) > 0
    assert (y[result["val_idx"]] != -1).all()  # no unlabeled in val


def test_split_covers_all_rows(tmp_path):
    y = _y()
    out = tmp_path / "s.npz"
    result = make_train_val_indices(y, out, seed=1, val_fraction=0.2)
    train_set = set(result["train_idx"])
    val_set = set(result["val_idx"])
    assert train_set.isdisjoint(val_set)
    assert len(train_set) + len(val_set) == len(y)


def test_load_split_indices_roundtrip(tmp_path):
    y = _y()
    out = tmp_path / "s.npz"
    make_train_val_indices(y, out, seed=5, val_fraction=0.1)
    loaded = load_split_indices(out)
    assert len(loaded["train_idx"]) + len(loaded["val_idx"]) == len(y)