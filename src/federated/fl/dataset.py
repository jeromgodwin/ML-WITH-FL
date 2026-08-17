"""Phase-4 partition consumer: per-client local datasets, nothing else.

Each client receives ONLY its own saved row indices (from the Phase 4
partition files) and materializes its local train/val arrays from the shared
vectorized matrix. Data never leaves the process — no sample data is ever
transmitted in the FL protocol (only model parameters are exchanged).

Scaling uses the Phase-3 fitted scaler (divide-by-std, zero-variance guard),
identical to the centralized baseline so FL results are directly comparable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.federated.data.partition import load_partition

logger = logging.getLogger(__name__)


class PartitionClientData:
    """Resolves Phase-4 indices into local per-client (X, y) float32 arrays."""

    def __init__(
        self,
        partition_dir: Path,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scale_inv: Optional[np.ndarray] = None,
        cache: bool = True,
    ):
        self.partition_dir = Path(partition_dir)
        self.config, self.pool, self.train_idx, self.val_idx, self.families = load_partition(
            self.partition_dir)
        self.X_train = X_train  # shared raw matrix (memmap or ndarray)
        self.y_train = y_train
        self.scale_inv = scale_inv
        self._cache: Dict[int, Any] = {}
        self._cache_enabled = cache
        if self.scale_inv is not None:
            self.scale_inv = np.asarray(self.scale_inv, dtype=np.float32)

    @property
    def n_clients(self) -> int:
        return self.config.clients

    def strategy(self) -> str:
        return self.config.strategy

    def client_sizes(self) -> Dict[int, Tuple[int, int]]:
        return {c: (len(self.train_idx[c]), len(self.val_idx[c]))
                for c in range(self.n_clients)}

    def client_data(self, cid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(X_train, y_train, X_val, y_val) for one client — its rows only."""
        if cid in self._cache:
            return self._cache[cid]
        tr = self.train_idx[cid]
        va = self.val_idx[cid]
        X_tr = np.asarray(self.X_train[tr], dtype=np.float32)
        y_tr = np.asarray(self.y_train[tr], dtype=np.int64)
        X_va = np.asarray(self.X_train[va], dtype=np.float32)
        y_va = np.asarray(self.y_train[va], dtype=np.int64)
        if self.scale_inv is not None:
            X_tr = X_tr * self.scale_inv
            X_va = X_va * self.scale_inv
        data = (X_tr, y_tr, X_va, y_va)
        if self._cache_enabled:
            self._cache[cid] = data
        return data

    def test_set(self, X_test: np.ndarray, y_test: np.ndarray):
        """Server-side global test accessor (kept as given; chunked at eval)."""
        return X_test, y_test