"""Server-side model validation gate (Phase 14).

Maintains a controlled validation dataset (rows never used by any client).
Pipeline per round:

    candidate aggregated model
    → validation on the held-out set
    → compare with the currently trusted model
    → accept / flag / reject

If the candidate's validation F1 degrades unexpectedly (below trusted F1
minus a tolerance), the candidate is REJECTED and the previous trusted
model is retained for the next round.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from fedshield.logging_setup import get_logger
from src.federated.evaluation.metrics import compute_metrics, predict_proba_chunked
from src.federated.models.mlp import MLPConfig, build_mlp

logger = get_logger(__name__)


class ValidationDecision(str, Enum):
    ACCEPT = "ACCEPT"
    FLAG = "FLAG"
    REJECT = "REJECT"


@dataclass
class ValidationRecord:
    """One round's validation measurement."""

    round: int
    candidate_f1: float
    trusted_f1: float
    delta_f1: float
    decision: ValidationDecision

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "candidate_f1": round(float(self.candidate_f1), 6),
            "trusted_f1": None if self.trusted_f1 is None
            else round(float(self.trusted_f1), 6),
            "delta_f1": round(float(self.delta_f1), 6),
            "decision": self.decision.value,
        }


class ValidationGate:
    """Validates candidate aggregated parameters against the trusted model.

    ``tolerance``: candidate F1 below ``trusted_f1 - tolerance`` is rejected.
    ``flag_delta``: candidates that improve or match the trusted model are
    accepted; anything within ``(trusted - tolerance, trusted + flag_delta)``
    is flagged (accepted but noted) unless it drops below the reject bar.
    """

    def __init__(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        scale_inv: Optional[np.ndarray],
        model_cfg: MLPConfig,
        tolerance: float = 0.01,
        flag_delta: float = 0.02,
        chunk: int = 20000,
    ):
        if len(X_val) != len(y_val):
            raise ValueError("validation features/labels length mismatch")
        self.X_val = np.asarray(X_val, dtype=np.float32)
        self.y_val = np.asarray(y_val)
        self.scale_inv = scale_inv
        self.model_cfg = model_cfg
        self.tolerance = tolerance
        self.flag_delta = flag_delta
        self.chunk = chunk
        self.trusted_params: Optional[Sequence[np.ndarray]] = None
        self.trusted_f1: Optional[float] = None
        self.records: List[ValidationRecord] = []
        self.n_rejects = 0
        self.n_flags = 0

    def _evaluate_f1(self, parameters: Sequence[np.ndarray]) -> float:
        model = build_mlp(self.model_cfg)
        with torch.no_grad():
            for p, new_p in zip(model.parameters(), parameters):
                p.copy_(torch.from_numpy(np.asarray(new_p, dtype=np.float32)))
        model.eval()
        probs = predict_proba_chunked(model, self.X_val, chunk=self.chunk,
                                      scale_inv=self.scale_inv)
        metrics = compute_metrics(self.y_val, probs)
        f1 = float(metrics["f1"])
        logger.info("validation gate: candidate f1=%.4f trusted f1=%s",
                    f1, "None" if self.trusted_f1 is None else f"{self.trusted_f1:.4f}")
        return f1

    def validate(
        self,
        server_round: int,
        candidate: Sequence[np.ndarray],
    ) -> ValidationRecord:
        """Validate candidate parameters; update the trusted model on accept."""
        candidate_f1 = self._evaluate_f1(candidate)
        trusted_f1 = self.trusted_f1
        if trusted_f1 is None:
            decision = ValidationDecision.ACCEPT
            self.trusted_params = [np.asarray(p, dtype=np.float32).copy()
                                   for p in candidate]
            self.trusted_f1 = candidate_f1
        else:
            delta = candidate_f1 - trusted_f1
            if candidate_f1 < trusted_f1 - self.tolerance:
                decision = ValidationDecision.REJECT
                self.n_rejects += 1
                logger.warning("validation gate round %d: REJECT candidate "
                               "(f1 %.4f < trusted %.4f - tol %.4f)",
                               server_round, candidate_f1, trusted_f1,
                               self.tolerance)
            elif delta <= self.flag_delta:
                decision = ValidationDecision.FLAG
                self.n_flags += 1
                self.trusted_params = [np.asarray(p, dtype=np.float32).copy()
                                       for p in candidate]
                self.trusted_f1 = candidate_f1
            else:
                decision = ValidationDecision.ACCEPT
                self.trusted_params = [np.asarray(p, dtype=np.float32).copy()
                                       for p in candidate]
                self.trusted_f1 = candidate_f1

        rec = ValidationRecord(round=server_round, candidate_f1=candidate_f1,
                               trusted_f1=trusted_f1,
                               delta_f1=candidate_f1 - (trusted_f1 or candidate_f1),
                               decision=decision)
        self.records.append(rec)
        return rec

    def trusted_parameters(self) -> Optional[Sequence[np.ndarray]]:
        """Currently trusted parameters (last accepted candidate)."""
        return self.trusted_params

    def summary(self) -> dict:
        return {
            "n_validations": len(self.records),
            "n_accept": sum(1 for r in self.records
                            if r.decision == ValidationDecision.ACCEPT),
            "n_flag": self.n_flags,
            "n_reject": self.n_rejects,
            "trusted_f1": None if self.trusted_f1 is None
            else round(float(self.trusted_f1), 6),
        }
