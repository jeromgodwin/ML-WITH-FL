"""Flower client: trains/evaluates on LOCAL data only (Phase 9, Phase 10, Phase 11).

A genuine flwr NumPyClient: receives the global model, trains locally on its
own rows, evaluates on its own holdout, and returns parameters + metrics.
The client also MEASURES its actual serialized payload sizes (upload =
parameters it returns, download = parameters it received) — real values, not
estimates.

FedProx (Phase 10): the client applies the proximal regularizer
    L_prox = L_local + (mu / 2) * ||w - w_global||^2
when the server sends ``proximal_mu > 0`` in the fit config. ``w_global`` is a
frozen snapshot of the global model taken at the START of the local round and
is never updated during local training. ``mu = 0`` degenerates to plain FedAvg
exactly (no regularizer), so the same client serves both strategies.

Personalized FL (Phase 11): ``PersonalizedClient`` (FedPer-style) keeps a
PERSONAL head (final layer) and only exchanges the shared body with the
server. The head is created once per client, is never transmitted, and is
never overwritten by received parameters — one client's personalized state can
never leak into another's.
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from flwr.client import NumPyClient

from src.federated.data.partition import ClientPartitionConfig  # noqa: F401
from src.federated.defense.attack import (
    apply_attack,
    flip_labels,
    is_malicious_cid,
)
from src.federated.evaluation.metrics import compute_metrics
from src.federated.fl.dataset import PartitionClientData
from src.federated.models.mlp import (
    MLPConfig, build_mlp, build_personalized_mlp,
)

device = "cpu"


def serialize_bytes(parameters: List[np.ndarray]) -> int:
    """Actual serialized size of model parameters (sum of raw tensor bytes)."""
    return int(sum(p.nbytes for p in parameters))


def bce_loss(logits: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(-(y * np.log(np.clip(logits, 1e-7, 1 - 1e-7))
                           + (1 - y) * np.log(np.clip(1 - logits, 1e-7, 1 - 1e-7)))))


class FedAvgClient(NumPyClient):
    def __init__(
        self,
        model_cfg: MLPConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        seed: int = 42,
        controller: Optional[Any] = None,
        cid: int = 0,
        attack_spec: Optional[Any] = None,
    ):
        self.model_cfg = model_cfg
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.seed = seed
        self.controller = controller  # Phase 12 resource gate (optional)
        self.cid = cid
        self.attack_spec = attack_spec  # Phase 14 simulated attack (optional)
        self.model = build_mlp(model_cfg)

    # ------------------------------------------------------------------
    def _training_gate(self) -> bool:
        """Block until the resource policy permits training (Phase 12).

        Returns False only when training was CANCELLED (the fit must stop).
        With no controller attached, training is always permitted. The gate
        is checked between epochs, never inside the real-time detection
        pipeline, which does not consult the controller at all.
        """
        if self.controller is None:
            return True
        t0 = time.perf_counter()
        allowed = self.controller.wait_until_allowed()
        self._gate_wait_ms += (time.perf_counter() - t0) * 1000.0
        return allowed

    # ------------------------------------------------------------------
    def get_parameters(self, config) -> List[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        download_bytes = serialize_bytes(parameters)
        self._set_parameters(parameters)
        # Frozen global reference for FedProx: captured once at round start,
        # never updated during local training. mu=0 makes the term vanish.
        w_global = [p.detach().clone() for p in self.model.parameters()]
        mu = float(config.get("proximal_mu", 0.0))

        # Phase 14 simulated attack (label_flip alters local training data).
        y_train = self.y_train
        attack_type = "none"
        if self.attack_spec is not None and is_malicious_cid(self.cid, self.attack_spec):
            attack_type = self.attack_spec.attack_type
            if attack_type == "label_flip":
                y_train = flip_labels(y_train, self.attack_spec.flip_frac,
                                      self.attack_spec.seed + self.cid)
        self.model.train()
        torch.manual_seed(self.seed + int(config.get("server_round", 0)) * 1000)

        lr = float(config.get("lr", 1e-3))
        local_epochs = int(config.get("local_epochs", 1))
        batch_size = int(config.get("batch_size", 512))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        n = self.X_train.shape[0]
        rng = np.random.default_rng(self.seed)
        t0 = time.perf_counter()
        self._gate_wait_ms = 0.0
        bce_terms, prox_terms, total_correct = 0.0, 0.0, 0
        epochs_completed = 0
        aborted = False
        for _ in range(local_epochs):
            if not self._training_gate():
                aborted = True
                break
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb = torch.from_numpy(self.X_train[idx])
                yb = torch.from_numpy(y_train[idx]).float()
                optimizer.zero_grad()
                logits = self.model(xb).ravel()
                bce = nn.functional.binary_cross_entropy_with_logits(logits, yb)
                loss = bce
                if mu > 0:
                    prox = (mu / 2.0) * sum(
                        (p - w).pow(2).sum()
                        for p, w in zip(self.model.parameters(), w_global))
                    loss = bce + prox
                    prox_terms += float(prox.detach().item())
                loss.backward()
                optimizer.step()
                bce_terms += float(bce.item()) * len(idx)
                total_correct += ((torch.sigmoid(logits) >= 0.5).float() == yb).sum().item()
            epochs_completed += 1
        fit_time_ms = (time.perf_counter() - t0) * 1000.0

        params = self.get_parameters({})
        # Phase 14 simulated attack (scaled_update / replacement) transforms
        # the returned parameters after honest training.
        if self.attack_spec is not None and is_malicious_cid(self.cid, self.attack_spec):
            params = apply_attack(params, list(parameters), self.cid, self.attack_spec)
        upload_bytes = serialize_bytes(params)
        metrics = {
            "train_loss": round(bce_terms / n, 6),
            "train_accuracy": round(total_correct / n, 6),
            "fit_time_ms": round(fit_time_ms, 2),
            "download_bytes": download_bytes,
            "upload_bytes": upload_bytes,
            "proximal_mu": mu,
            "prox_penalty": round(prox_terms, 6),
            "partition_cid": self.cid,
        }
        if self.attack_spec is not None and is_malicious_cid(self.cid, self.attack_spec):
            metrics["simulated_attack"] = attack_type
        if self.controller is not None:
            metrics["resource_gated"] = True
            metrics["epochs_completed"] = epochs_completed
            metrics["gate_wait_ms"] = round(self._gate_wait_ms, 2)
            metrics["aborted"] = aborted
        return params, n, metrics

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, Any]]:
        self._set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(self.X_val)).numpy().ravel()
        probs = 1.0 / (1.0 + np.exp(-logits))
        loss = bce_loss(probs, self.y_val)
        m = compute_metrics(self.y_val, probs)
        metrics = {k: m[k] for k in ("accuracy", "precision", "recall", "f1")
                   if m.get(k) is not None}
        if m.get("roc_auc") is not None:
            metrics["roc_auc"] = m["roc_auc"]
        metrics["n_val"] = int(len(self.y_val))
        return loss, len(self.y_val), metrics

    # ------------------------------------------------------------------
    def _set_parameters(self, parameters: List[np.ndarray]) -> None:
        with torch.no_grad():
            for p, new_p in zip(self.model.parameters(), parameters):
                p.copy_(torch.from_numpy(new_p))


class PersonalizedClient(FedAvgClient):
    """FedPer/FedRep-style personalized client: shared BODY + personal HEAD.

    Phase 11. The client keeps a persistent personal head (final layer) and
    trains FedRep-style (Collins et al. 2021): each round it FIRST adapts the
    personal head to the current global body with the body frozen (head
    epochs, ``head_lr``), THEN trains the shared body with the head frozen
    (``local_epochs`` at ``lr``, same budget as FedAvg's body training). The
    decoupling prevents head/body co-adaptation resonance, which otherwise
    makes the model flip between degenerate all-positive/all-negative fixed
    points on real EMBER data.

    Only the body crosses the wire:

    - the head is created once when the client is constructed (client-specific
      state), is NEVER transmitted (upload/download bytes exclude it), and is
      NEVER overwritten by received parameters (``_set_parameters`` only
      touches the body), so one client's head can never be replaced by
      another's or by the server's
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = build_personalized_mlp(self.model_cfg)

    def get_parameters(self, config) -> List[np.ndarray]:
        """Return ONLY the shared body parameters (head stays local)."""
        return [p.detach().cpu().numpy() for p in self.model.body.parameters()]

    def _set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load global body parameters; the personal head is left untouched."""
        with torch.no_grad():
            for p, new_p in zip(self.model.body.parameters(), parameters):
                p.copy_(torch.from_numpy(new_p))

    def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        download_bytes = serialize_bytes(parameters)
        self._set_parameters(parameters)
        self.model.train()
        # Dropout masks come from per-client LOCAL generators (DeterministicDropout),
        # seeded identically per round: concurrent client threads cannot corrupt
        # each other's mask streams through the shared global RNG.
        self.model.set_dropout_seed(self.seed + int(config.get("server_round", 0)) * 1000)

        lr = float(config.get("lr", 1e-3))
        local_epochs = int(config.get("local_epochs", 1))
        batch_size = int(config.get("batch_size", 512))
        head_epochs = int(config.get("head_epochs", 2))
        head_lr = float(config.get("head_lr", 1e-2))

        n = self.X_train.shape[0]
        rng = np.random.default_rng(self.seed)
        t0 = time.perf_counter()
        self._gate_wait_ms = 0.0

        # Phase A: adapt the personal head to the current global body.
        for p in self.model.body.parameters():
            p.requires_grad = False
        head_opt = torch.optim.Adam(self.model.head.parameters(), lr=head_lr)
        head_loss_terms = 0.0
        head_epochs_completed = 0
        aborted = False
        for _ in range(head_epochs):
            if not self._training_gate():
                aborted = True
                break
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb = torch.from_numpy(self.X_train[idx])
                yb = torch.from_numpy(self.y_train[idx]).float()
                head_opt.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(
                    self.model(xb).ravel(), yb)
                loss.backward()
                head_opt.step()
                head_loss_terms += float(loss.item()) * len(idx)
            head_epochs_completed += 1

        # Phase B: train the shared body with the head frozen.
        for p in self.model.body.parameters():
            p.requires_grad = True
        for p in self.model.head.parameters():
            p.requires_grad = False
        body_opt = torch.optim.Adam(self.model.body.parameters(), lr=lr)
        bce_terms, total_correct = 0.0, 0
        body_epochs_completed = 0
        for _ in range(local_epochs):
            if not self._training_gate():
                aborted = True
                break
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb = torch.from_numpy(self.X_train[idx])
                yb = torch.from_numpy(self.y_train[idx]).float()
                body_opt.zero_grad()
                logits = self.model(xb).ravel()
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
                loss.backward()
                body_opt.step()
                bce_terms += float(loss.item()) * len(idx)
                total_correct += ((torch.sigmoid(logits) >= 0.5).float() == yb).sum().item()
            body_epochs_completed += 1
        for p in self.model.head.parameters():
            p.requires_grad = True
        fit_time_ms = (time.perf_counter() - t0) * 1000.0

        params = self.get_parameters({})
        upload_bytes = serialize_bytes(params)
        metrics = {
            "train_loss": round(bce_terms / n, 6),
            "head_train_loss": round(head_loss_terms / n, 6),
            "train_accuracy": round(total_correct / n, 6),
            "fit_time_ms": round(fit_time_ms, 2),
            "download_bytes": download_bytes,
            "upload_bytes": upload_bytes,
            "proximal_mu": 0.0,
            "prox_penalty": 0.0,
        }
        if self.controller is not None:
            metrics["resource_gated"] = True
            metrics["head_epochs_completed"] = head_epochs_completed
            metrics["body_epochs_completed"] = body_epochs_completed
            metrics["gate_wait_ms"] = round(self._gate_wait_ms, 2)
            metrics["aborted"] = aborted
        return params, n, metrics

    @property
    def head_params(self) -> List[np.ndarray]:
        """The personal head as numpy arrays (for tests/inspection only)."""
        return [p.detach().cpu().numpy() for p in self.model.head.parameters()]


def build_client_fn(
    data: PartitionClientData,
    model_cfg: MLPConfig,
    seed: int = 42,
    cid: Optional[int] = None,
    controller: Optional[Any] = None,
    attack_spec: Optional[Any] = None,
) -> Any:
    """client_fn(cid) factory: each client materializes ONLY its own rows.

    cid: the partition index this worker is bound to. If None, it falls back
    to the ``cid`` string Flower passes in. IMPORTANT: in-process workers
    started via flwr.client.start_client all receive the SAME node id
    (gRPC-bidi has no node concept), so callers MUST pass the explicit
    partition index to keep clients on their own data.

    controller: optional Phase-12 TrainingController shared by all clients;
    the client gates each epoch through it (None = unrestricted training).

    attack_spec: optional Phase-14 AttackSpec; the first ``n_malicious``
    clients simulate the configured abnormal-update attack (synthetic only).
    """

    def client_fn(cid_str: str):
        index = int(cid) if cid is not None else int(cid_str)
        X_tr, y_tr, X_va, y_va = data.client_data(index)
        return FedAvgClient(model_cfg, X_tr, y_tr, X_va, y_va, seed=seed,
                            controller=controller, cid=index,
                            attack_spec=attack_spec).to_client()

    return client_fn


def build_personalized_client_fn(
    data: PartitionClientData,
    model_cfg: MLPConfig,
    seed: int = 42,
    cid: Optional[int] = None,
    controller: Optional[Any] = None,
    attack_spec: Optional[Any] = None,
) -> Any:
    """client_fn(cid) factory for PersonalizedClient (Phase 11).

    Same worker-to-partition binding contract as ``build_client_fn``; each
    client additionally owns a personal head. The head's initialization is
    deterministic: model construction is serialized under a lock while the
    global RNG is reseeded, so every client builds from the same RNG state
    (all heads start identical and diverge purely through local data) and the
    result is reproducible regardless of thread scheduling.

    controller: optional Phase-12 TrainingController shared by all clients.
    attack_spec: optional Phase-14 AttackSpec (synthetic abnormal updates).
    """

    def client_fn(cid_str: str):
        index = int(cid) if cid is not None else int(cid_str)
        X_tr, y_tr, X_va, y_va = data.client_data(index)
        from src.utils.reproducibility import set_all_seeds
        with _model_build_lock:
            set_all_seeds(seed)
            client = PersonalizedClient(
                model_cfg, X_tr, y_tr, X_va, y_va, seed=seed,
                controller=controller, cid=index, attack_spec=attack_spec)
        return client.to_client()

    return client_fn


_model_build_lock = Lock()