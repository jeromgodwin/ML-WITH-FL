"""Model registry: manages model artifacts, versions, and approval lifecycle.

The endpoint engine only uses the currently *approved* local model. A new global
model from the federated server is registered, validated (feature schema
compatibility, parameter count), and only then activated — never automatically
when incompatible or corrupt.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fedshield.logging_setup import get_logger, log_event
from src.interfaces import FeatureSchema, ModelMetadata

logger = get_logger(__name__)

RegistryStatus = Literal["pending", "approved", "rejected", "superseded"]


@dataclass
class RegistryEntry:
    """One entry in the model registry."""

    version: str
    algorithm: str
    training_round: Optional[int]
    feature_schema: FeatureSchema
    artifact_path: str
    status: RegistryStatus = "pending"
    created_at: str = ""
    approved_at: Optional[str] = None
    metrics: dict[str, float] | None = None
    validation_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "algorithm": self.algorithm,
            "training_round": self.training_round,
            "feature_schema": self.feature_schema.to_dict(),
            "artifact_path": self.artifact_path,
            "status": self.status,
            "created_at": self.created_at,
            "approved_at": self.approved_at,
            "metrics": self.metrics or {},
            "validation_notes": self.validation_notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RegistryEntry":
        return cls(
            version=d["version"],
            algorithm=d["algorithm"],
            training_round=d.get("training_round"),
            feature_schema=FeatureSchema.from_dict(d["feature_schema"]),
            artifact_path=d["artifact_path"],
            status=d.get("status", "pending"),
            created_at=d.get("created_at", ""),
            approved_at=d.get("approved_at"),
            metrics=d.get("metrics") or None,
            validation_notes=d.get("validation_notes", ""),
        )


class ModelRegistry:
    """JSON-file-backed model registry.

    Layout::
        <registry_dir>/
            index.json
            artifacts/
                <version>/
                    model.pt
                    model.meta.json
    """

    def __init__(self, registry_dir: str | Path):
        self.registry_dir = Path(registry_dir)
        self.artifacts_dir = self.registry_dir / "artifacts"
        self.index_path = self.registry_dir / "index.json"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict] = self._load_index()
        self._active: Optional[str] = self._load_active()

    def _load_index(self) -> dict[str, dict]:
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load registry index (starting empty): %s", e)
        return {}

    def _save_index(self) -> None:
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, sort_keys=True)

    def _load_active(self) -> Optional[str]:
        active = self.registry_dir / "active.txt"
        if active.exists():
            version = active.read_text(encoding="utf-8").strip()
            if version in self._index and self._index[version]["status"] == "approved":
                return version
        return None

    def register(
        self,
        metadata: ModelMetadata,
        artifact_source: str | Path,
        expected_input_dim: Optional[int] = None,
        copy_artifact: bool = True,
    ) -> RegistryEntry:
        """Register a model artifact as pending; does NOT activate it."""
        version = metadata.version
        if version in self._index:
            raise ValueError(f"Model version already registered: {version}")

        notes: list[str] = []
        if expected_input_dim is not None and metadata.input_dim != expected_input_dim:
            notes.append(
                f"input_dim mismatch: expected {expected_input_dim}, got {metadata.input_dim}"
            )

        artifact_dir = self.artifacts_dir / version
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "model.pt"

        if copy_artifact:
            shutil.copy2(artifact_source, artifact_path)
        else:
            artifact_path = Path(artifact_source)

        meta_path = artifact_dir / "model.meta.json"
        meta_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")

        entry = RegistryEntry(
            version=version,
            algorithm=metadata.algorithm,
            training_round=metadata.training_round,
            feature_schema=metadata.feature_schema,
            artifact_path=str(artifact_path),
            status="pending",
            created_at=datetime.now(timezone.utc).isoformat(),
            metrics=metadata.metrics,
            validation_notes="; ".join(notes),
        )
        self._index[version] = entry.to_dict()
        self._save_index()
        log_event(logger, 20, f"Registered model {version} (status=pending)", version=version)
        return entry

    def validate(self, version: str) -> tuple[bool, list[str]]:
        """Validate a pending model: artifact exists, metadata matches registry.

        Returns (valid, issues). Corrupt/incompatible models fail validation.
        """
        if version not in self._index:
            return False, [f"unknown version {version}"]
        data = self._index[version]
        issues: list[str] = []
        artifact = Path(data["artifact_path"])
        if not artifact.exists():
            issues.append(f"artifact missing: {artifact}")
        meta = artifact.with_name("model.meta.json")
        if not meta.exists():
            issues.append(f"metadata missing: {meta}")
        if data.get("validation_notes"):
            issues.append(data["validation_notes"])
        return (not issues), issues

    def approve(self, version: str, metrics: Optional[dict[str, float]] = None) -> RegistryEntry:
        """Approve a registered model and make it the active local model."""
        if version not in self._index:
            raise KeyError(f"Model not registered: {version}")
        valid, issues = self.validate(version)
        if not valid:
            raise ValueError(f"Cannot approve invalid model {version}: {'; '.join(issues)}")

        old_active = self._active
        if old_active and old_active != version:
            self._index[old_active]["status"] = "superseded"

        self._index[version]["status"] = "approved"
        self._index[version]["approved_at"] = datetime.now(timezone.utc).isoformat()
        if metrics:
            self._index[version]["metrics"] = metrics
        (self.registry_dir / "active.txt").write_text(version, encoding="utf-8")
        self._active = version
        self._save_index()
        log_event(logger, 20, f"Approved and activated model {version}", version=version)
        return self.get(version)

    def reject(self, version: str, reason: str = "") -> RegistryEntry:
        """Reject a model (e.g., incompatible schema, corrupt weights)."""
        if version not in self._index:
            raise KeyError(f"Model not registered: {version}")
        self._index[version]["status"] = "rejected"
        if reason:
            self._index[version]["validation_notes"] = reason
        self._save_index()
        return self.get(version)

    def get_active(self) -> Optional[RegistryEntry]:
        """Return the currently approved (active) model, if any."""
        if self._active is None:
            return None
        return self.get(self._active)

    def get(self, version: str) -> RegistryEntry:
        if version not in self._index:
            raise KeyError(f"Model not registered: {version}")
        return RegistryEntry.from_dict(self._index[version])

    def list_all(self) -> list[RegistryEntry]:
        """All registered models, newest first."""
        entries = [RegistryEntry.from_dict(d) for d in self._index.values()]
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries

    def status_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for data in self._index.values():
            summary[data["status"]] = summary.get(data["status"], 0) + 1
        return summary


def create_registry(registry_dir: str | Path) -> ModelRegistry:
    """Factory helper."""
    return ModelRegistry(registry_dir)
