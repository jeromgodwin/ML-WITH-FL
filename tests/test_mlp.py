"""MLP model tests: shapes, determinism, parameter counts."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.federated.models.mlp import (  # noqa: E402
    MLPConfig, build_mlp, count_flops, count_parameters, init_model, model_size_bytes,
)


def test_forward_shape():
    cfg = MLPConfig(input_dim=64, hidden_layers=(32, 16), dropout=0.2)
    model = build_mlp(cfg)
    x = torch.randn(8, 64)
    out = model(x)
    assert out.shape == (8, 1)


def test_output_dim_is_single_logit():
    cfg = MLPConfig(input_dim=10, hidden_layers=(8,))
    model = build_mlp(cfg)
    assert model(torch.randn(3, 10)).shape == (3, 1)


def test_eval_disables_dropout():
    cfg = MLPConfig(input_dim=16, hidden_layers=(16, 8), dropout=0.5)
    model = build_mlp(cfg)
    model.eval()
    x = torch.randn(64, 16)
    a = model(x)
    b = model(x)
    assert torch.allclose(a, b)


def test_parameter_count():
    cfg = MLPConfig(input_dim=10, hidden_layers=(20, 10), dropout=0.0)
    model = build_mlp(cfg)
    expected = 10 * 20 + 20 + 20 * 10 + 10 + 10 * 1 + 1
    assert count_parameters(model) == expected
    assert model_size_bytes(model) > 0


def test_seed_determinism():
    cfg = MLPConfig(input_dim=32, hidden_layers=(16,))
    m1 = init_model(cfg, seed=7)
    m2 = init_model(cfg, seed=7)
    m3 = init_model(cfg, seed=8)
    for (k1, v1), (k2, v2), (k3, v3) in zip(
            m1.state_dict().items(), m2.state_dict().items(), m3.state_dict().items()):
        assert torch.equal(v1, v2)
        assert not torch.equal(v1, v3)


def test_count_flops_positive():
    cfg = MLPConfig(input_dim=64, hidden_layers=(32,))
    assert count_flops(cfg, batch=10) > 0


def test_unsupported_activation():
    cfg = MLPConfig(input_dim=4, hidden_layers=(4,), activation="bogus")
    with pytest.raises(ValueError):
        build_mlp(cfg)