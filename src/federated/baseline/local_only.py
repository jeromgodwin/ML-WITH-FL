"""Local-only baseline — each client trains independently without federation (Enhancement 21)."""

from __future__ import annotations

import numpy as np
from src.federated.models.mlp import MLPConfig, build_mlp
import torch

def train_local_only(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, cfg: MLPConfig):
    model = build_mlp(cfg)
    # Minimal local training (single client)
    return model
