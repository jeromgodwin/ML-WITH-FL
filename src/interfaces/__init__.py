"""Shared interfaces between Endpoint Protection Engine and Federated Learning Engine.

These define the contract for model artifacts, feature schemas, and detection records
that flow between the two subsystems.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class FeatureSchema:
    """Feature schema that must match between training and inference.

    This is stored alongside the model and validated at load time.
    """

    feature_names: tuple[str, ...]
    feature_types: tuple[str, ...]  # e.g., "float32", "int64"
    preprocessing_version: str
    created_at: str  # ISO timestamp
    model_version: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureSchema":
        return cls(
            feature_names=tuple(d["feature_names"]),
            feature_types=tuple(d["feature_types"]),
            preprocessing_version=d["preprocessing_version"],
            created_at=d["created_at"],
            model_version=d["model_version"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "feature_types": list(self.feature_types),
            "preprocessing_version": self.preprocessing_version,
            "created_at": self.created_at,
            "model_version": self.model_version,
        }


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata for a trained model (local or federated)."""

    version: str
    algorithm: str  # fedavg | fedprox | personalized | centralized
    training_round: Optional[int]
    feature_schema: FeatureSchema
    metrics: dict[str, float]  # accuracy, f1, etc.
    created_at: str  # ISO timestamp
    num_parameters: int
    input_dim: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelMetadata":
        return cls(
            version=d["version"],
            algorithm=d["algorithm"],
            training_round=d.get("training_round"),
            feature_schema=FeatureSchema.from_dict(d["feature_schema"]),
            metrics=d.get("metrics", {}),
            created_at=d["created_at"],
            num_parameters=d["num_parameters"],
            input_dim=d["input_dim"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "algorithm": self.algorithm,
            "training_round": self.training_round,
            "feature_schema": self.feature_schema.to_dict(),
            "metrics": self.metrics,
            "created_at": self.created_at,
            "num_parameters": self.num_parameters,
            "input_dim": self.input_dim,
        }


@dataclass(frozen=True)
class DetectionRecord:
    """Record of a single file analysis (for history/notifications).

    Phase 7: verdict is a policy level (LOW | MEDIUM | HIGH), action is the
    policy decision (ALLOW | WARN | QUARANTINE). Thresholds that map
    probability -> verdict/action are configuration, not universal truth.
    """

    detection_id: str  # unique per analysis
    timestamp: str  # ISO format
    filename: str
    filepath: str
    sha256: str
    file_type: str  # pe_exe | pe_dll | pe_sys | unknown | unsupported
    model_version: str
    malware_probability: float
    benign_probability: Optional[float]
    risk_score: int  # 0-100
    risk_level: str  # LOW | MEDIUM | HIGH
    verdict: Literal["LOW", "MEDIUM", "HIGH", "ERROR"]
    action: Literal["ALLOW", "WARN", "QUARANTINE", "NONE"]
    model_algorithm: str
    analysis_duration_ms: float = 0.0  # total scan time

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DetectionRecord":
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "timestamp": self.timestamp,
            "filename": self.filename,
            "filepath": self.filepath,
            "sha256": self.sha256,
            "file_type": self.file_type,
            "model_version": self.model_version,
            "malware_probability": self.malware_probability,
            "benign_probability": self.benign_probability,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "verdict": self.verdict,
            "action": self.action,
            "model_algorithm": self.model_algorithm,
            "analysis_duration_ms": self.analysis_duration_ms,
        }


@dataclass(frozen=True)
class QuarantineRecord:
    """Record of a quarantined file."""

    original_path: str
    quarantine_path: str
    sha256: str
    detection: DetectionRecord
    quarantined_at: str  # ISO timestamp
    restored: bool = False
    restored_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_path": self.original_path,
            "quarantine_path": self.quarantine_path,
            "sha256": self.sha256,
            "detection": self.detection.to_dict(),
            "quarantined_at": self.quarantined_at,
            "restored": self.restored,
            "restored_at": self.restored_at,
        }
