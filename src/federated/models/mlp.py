"""Small configurable MLP for EMBER static PE features."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import torch
from torch import nn


@dataclass
class MLPConfig:
    input_dim: int = 2381
    hidden_layers: Tuple[int, ...] = (256, 128)
    dropout: float = 0.2
    activation: str = "relu"

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_layers": list(self.hidden_layers),
            "dropout": self.dropout,
            "activation": self.activation,
        }


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unsupported activation: {name}")


class DeterministicDropout(nn.Module):
    """Dropout whose masks come from a LOCAL generator, not the global RNG.

    Personalized FL runs its clients in concurrent threads; if dropout masks
    were drawn from the global torch RNG (as nn.Dropout does), the per-client
    mask stream would depend on thread interleaving, making training
    non-reproducible across runs (observed: identical configs produced
    different per-round client metrics). ``set_seed`` re-seeds the local
    generator deterministically per fit round, so every client always draws
    the exact same masks for a given (seed, round), regardless of scheduling.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = float(p)
        self.generator = torch.Generator()

    def set_seed(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        mask = torch.bernoulli(torch.full_like(x, 1.0 - self.p),
                               generator=self.generator)
        return x * mask / (1.0 - self.p)


class BinaryMLP(nn.Module):
    """Binary classifier: input -> [Linear-ReLU-Dropout]* -> Linear(1)."""

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        dims = (config.input_dim,) + tuple(config.hidden_layers)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(_activation(config.activation))
            if config.dropout > 0 and i < len(dims) - 2:
                layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (N, 1)."""
        return self.net(x)


class PersonalizedMLP(nn.Module):
    """BinaryMLP split into a shared BODY + a personal HEAD (Phase 11).

    Phase 11 (personalized FL): the body (all hidden layers) is the shared
    global representation, aggregated by the server every round. The head
    (final Linear(dims[-1], 1)) is CLIENT-SPECIFIC state: it is trained
    locally, never sent to the server, and never overwritten by received
    parameters. Forward is identical to BinaryMLP.

    The body output is standardized by a parameterless LayerNorm before the
    head: EMBER's long-tail raw-count features can produce body activations of
    magnitude ~1e6, which would saturate any head (logits >> 1) and stall its
    gradients. LayerNorm (no affine) keeps the head input at O(1) scale for
    every client and for the server-side probe head, at zero extra parameters.

    Dropout uses ``DeterministicDropout`` (local generator, seeded per fit
    round) so concurrent client threads cannot corrupt each other's masks via
    the global RNG: training is reproducible across runs.
    """

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        body: list[nn.Module] = []
        dims = (config.input_dim,) + tuple(config.hidden_layers)
        for i in range(len(dims) - 1):
            body.append(nn.Linear(dims[i], dims[i + 1]))
            body.append(_activation(config.activation))
            if config.dropout > 0 and i < len(dims) - 2:
                body.append(DeterministicDropout(config.dropout))
        self.body = nn.Sequential(*body)
        self._norm = nn.LayerNorm(dims[-1], elementwise_affine=False)
        self.head = nn.Sequential(self._norm, nn.Linear(dims[-1], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (N, 1)."""
        return self.head(self.body(x))

    def body_parameters(self):
        """Iterable of the shared (server-aggregated) parameters only."""
        return self.body.parameters()

    def set_dropout_seed(self, seed: int) -> None:
        """Seed every dropout module's local mask generator (deterministic)."""
        for module in self.body.modules():
            if isinstance(module, DeterministicDropout):
                module.set_seed(seed)


def build_mlp(config: MLPConfig) -> BinaryMLP:
    return BinaryMLP(config)


def build_personalized_mlp(config: MLPConfig) -> PersonalizedMLP:
    return PersonalizedMLP(config)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_bytes(model: nn.Module) -> int:
    """Approximate on-disk size of the state dict in bytes (float32 params)."""
    total = 0
    for tensor in model.state_dict().values():
        total += tensor.numel() * tensor.element_size()
    return total


def init_model(config: MLPConfig, seed: int) -> BinaryMLP:
    """Build and initialize a model deterministically for a given seed."""
    from src.utils.reproducibility import set_all_seeds
    set_all_seeds(seed)
    return build_mlp(config)


def count_flops(config: MLPConfig, batch: int = 1) -> float:
    """Approximate forward FLOPs for one batch (for training-time estimates)."""
    flops = 0.0
    dims = (config.input_dim,) + tuple(config.hidden_layers) + (1,)
    for i in range(len(dims) - 1):
        flops += 2.0 * batch * dims[i] * dims[i + 1]
    return flops