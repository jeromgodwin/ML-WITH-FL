"""Server-side defenses against malicious client updates (Phase 14).

Controlled cybersecurity experiment: simulate abnormal model updates and
measure attack impact / detection / mitigation. No real malware, no
operational attack tooling — synthetic abnormal updates only.
"""

from src.federated.defense.anomaly import (
    AnomalyClass,
    AnomalyRecord,
    UpdateAnomalyDetector,
)
from src.federated.defense.attack import (
    AttackSpec,
    apply_attack,
    attack_report,
    flip_labels,
    is_malicious_cid,
)
from src.federated.defense.clipping import (
    ClipRecord,
    UpdateClipper,
)
from src.federated.defense.robust import (
    coordinate_wise_median,
    trimmed_mean,
)
from src.federated.defense.validation import (
    ValidationDecision,
    ValidationRecord,
    ValidationGate,
)

__all__ = [
    "AnomalyClass",
    "AnomalyRecord",
    "UpdateAnomalyDetector",
    "AttackSpec",
    "apply_attack",
    "attack_report",
    "flip_labels",
    "is_malicious_cid",
    "ClipRecord",
    "UpdateClipper",
    "coordinate_wise_median",
    "trimmed_mean",
    "ValidationDecision",
    "ValidationRecord",
    "ValidationGate",
]
