"""Tests for reproducibility utilities: seeds, IDs, snapshots, path handling."""

import json
import random

import numpy as np

from src.utils.reproducibility import (
    config_fingerprint,
    generate_experiment_id,
    load_config_snapshot,
    resolve_project_path,
    save_config_snapshot,
    set_all_seeds,
)


def test_set_all_seeds_is_reproducible():
    set_all_seeds(7)
    a = [random.randint(0, 1000) for _ in range(5)]
    b = np.random.rand(3).tolist()
    set_all_seeds(7)
    assert [random.randint(0, 1000) for _ in range(5)] == a
    assert np.random.rand(3).tolist() == b


def test_experiment_id_unique_and_readable():
    ids = {generate_experiment_id("FedAvg Test") for _ in range(50)}
    assert len(ids) == 50
    first = generate_experiment_id("MyExp", seed=42)
    assert first.startswith("myexp-")
    assert first.endswith("-s42")


def test_config_fingerprint_deterministic():
    cfg_a = {"fl": {"num_rounds": 5, "mu": 0.1}, "seed": 42}
    cfg_b = {"seed": 42, "fl": {"mu": 0.1, "num_rounds": 5}}
    assert config_fingerprint(cfg_a) == config_fingerprint(cfg_b)
    cfg_c = {"fl": {"num_rounds": 6, "mu": 0.1}, "seed": 42}
    assert config_fingerprint(cfg_a) != config_fingerprint(cfg_c)


def test_config_snapshot_roundtrip(tmp_path):
    cfg = {"name": "exp1", "fl": {"num_rounds": 10}}
    out = save_config_snapshot(cfg, tmp_path)
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == cfg
    assert load_config_snapshot(out) == cfg


def test_resolve_project_path_absolute_and_tilde(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    absolute = resolve_project_path(str(tmp_path / "x"))
    assert absolute.is_absolute()

    home = resolve_project_path("~/data")
    assert home == (tmp_path / "data").resolve()


def test_resolve_project_path_is_root_relative():
    # A relative path must resolve inside the repo root (no machine-specific paths)
    resolved = resolve_project_path("models/local.pt")
    assert resolved.is_absolute()
    repo_root = resolved.parent.parent
    assert (repo_root / "pyproject.toml").exists()
