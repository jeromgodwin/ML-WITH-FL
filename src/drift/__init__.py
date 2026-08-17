"""Concept drift detection and adaptive retraining (Phase 13).

Public API:
    DriftConfig       configuration dataclass
    DriftDetector     PSI-based drift detector
    RetrainingSafety  safety guard (cooldown, frequency, min data)
    AdaptiveRetrainingManager  full workflow orchestration
"""

from fedshield.config import DriftConfig

from src.drift.detector import DriftDetector, compute_psi, DriftResult
from src.drift.manager import AdaptiveRetrainingManager, RetrainEvent
from src.drift.safety import RetrainingSafety, SafetyCheck

__all__ = [
    "DriftConfig", "DriftDetector", "compute_psi", "DriftResult",
    "RetrainingSafety", "SafetyCheck",
    "AdaptiveRetrainingManager", "RetrainEvent",
]