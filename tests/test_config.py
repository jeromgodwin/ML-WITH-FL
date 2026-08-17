"""Smoke tests for the Phase 1 scaffold: config loading and overrides."""

import copy

import pytest

from fedshield.config import ExperimentConfig, set_dotted


def test_load_default_yaml():
    cfg = ExperimentConfig.from_yaml("configs/default.yaml")
    assert cfg.fl.num_rounds == 30
    assert cfg.fl.algorithm == "fedavg"
    assert cfg.model.input_dim == 2381
    assert cfg.partition.strategy == "iid"
    # Endpoint config (YAML loads tuples as lists)
    assert cfg.endpoint.monitor.stability_wait == 2.0
    assert list(cfg.endpoint.risk.thresholds) == [0.3, 0.7]
    assert cfg.endpoint.quarantine.quarantine_dir == "quarantine"


def test_from_dict_keeps_defaults():
    cfg = ExperimentConfig.from_dict({"fl": {"num_rounds": 5}})
    assert cfg.fl.num_rounds == 5
    assert cfg.fl.num_clients == 10
    assert cfg.train.learning_rate == 0.001


def test_overrides_are_deep_copied():
    cfg = ExperimentConfig()
    cfg2 = cfg.with_overrides(**{"fl.num_rounds": 7, "train.seed": 7})
    assert cfg2.fl.num_rounds == 7
    assert cfg.fl.num_rounds == 30
    assert cfg2.train.seed == 7
    assert cfg.train.seed == 42


def test_endpoint_overrides():
    cfg = ExperimentConfig()
    cfg2 = cfg.with_overrides(**{"endpoint.monitor.stability_wait": 5.0, "endpoint.risk.thresholds": [0.2, 0.8]})
    assert cfg2.endpoint.monitor.stability_wait == 5.0
    assert cfg.endpoint.monitor.stability_wait == 2.0
    assert list(cfg2.endpoint.risk.thresholds) == [0.2, 0.8]


def test_to_dict_is_json_serializable():
    import json

    cfg = ExperimentConfig.from_yaml("configs/default.yaml")
    payload = json.dumps(cfg.to_dict())
    assert "fedavg" in payload


def test_set_dotted():
    cfg = ExperimentConfig()
    set_dotted(cfg, "fl.proximal_mu", 0.1)
    assert cfg.fl.proximal_mu == 0.1


def test_unknown_config_yaml_raises():
    with pytest.raises(FileNotFoundError):
        ExperimentConfig.from_yaml("does/not/exist.yaml")


def test_copy_is_independent():
    cfg = ExperimentConfig()
    other = copy.deepcopy(cfg)
    other.fl.num_clients = 99
    assert cfg.fl.num_clients == 10
