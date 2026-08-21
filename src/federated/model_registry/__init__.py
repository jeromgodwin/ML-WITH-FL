"""Model registry: manages model artifacts, validation, activation, rollback.

Phase 18 — Connects FL training to the automated endpoint client.

Flow: FL training → candidate model → validation → registry → approved model → client update → endpoint uses new model

Tracks: model ID, version, algorithm, training round, feature schema version,
preprocessing version, configuration, validation metrics, timestamp, status.

Statuses (canonical Phase 18): CANDIDATE, VALIDATED, ACTIVE, REJECTED, ARCHIVED
Legacy aliases supported: pending→CANDIDATE, approved→ACTIVE, rejected→REJECTED,
superseded→ARCHIVED (kept for backward compatibility with Phases 1-17).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from fedshield.logging_setup import get_logger, log_event
from src.interfaces import FeatureSchema, ModelMetadata

logger = get_logger(__name__)

# Canonical Phase 18 statuses + legacy aliases (all accepted on read/write)
RegistryStatus = Literal[
    "CANDIDATE", "VALIDATED", "ACTIVE", "REJECTED", "ARCHIVED",
    "pending", "approved", "rejected", "superseded",
]

# Mapping legacy → canonical
_LEGACY_TO_CANONICAL = {
    "pending": "CANDIDATE",
    "approved": "ACTIVE",
    "rejected": "REJECTED",
    "superseded": "ARCHIVED",
}
_CANONICAL_SET = {"CANDIDATE", "VALIDATED", "ACTIVE", "REJECTED", "ARCHIVED"}


def _canonical_status(s: str) -> str:
    return _LEGACY_TO_CANONICAL.get(s, s)


def _is_active_status(s: str) -> bool:
    return _canonical_status(s) == "ACTIVE"


def _is_candidate_status(s: str) -> bool:
    return _canonical_status(s) == "CANDIDATE"


@dataclass
class RegistryEntry:
    """One entry in the model registry (Phase 18)."""

    version: str
    algorithm: str
    training_round: Optional[int]
    feature_schema: FeatureSchema
    artifact_path: str
    status: RegistryStatus = "CANDIDATE"
    created_at: str = ""
    approved_at: Optional[str] = None
    validated_at: Optional[str] = None
    metrics: dict[str, float] | None = None
    validation_metrics: dict[str, float] | None = None
    validation_notes: str = ""
    # Phase 18 rich fields
    model_id: str = ""
    feature_schema_version: str = ""
    preprocessing_version: str = ""
    configuration: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    integrity_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "algorithm": self.algorithm,
            "training_round": self.training_round,
            "feature_schema": self.feature_schema.to_dict(),
            "feature_schema_version": self.feature_schema_version,
            "preprocessing_version": self.preprocessing_version,
            "configuration": self.configuration,
            "validation_metrics": self.validation_metrics or {},
            "artifact_path": self.artifact_path,
            "status": self.status,
            "created_at": self.created_at,
            "timestamp": self.timestamp or self.created_at,
            "approved_at": self.approved_at,
            "validated_at": self.validated_at,
            "metrics": self.metrics or {},
            "validation_notes": self.validation_notes,
            "integrity_hash": self.integrity_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RegistryEntry":
        return cls(
            model_id=d.get("model_id", d.get("version", "")),
            version=d["version"],
            algorithm=d["algorithm"],
            training_round=d.get("training_round"),
            feature_schema=FeatureSchema.from_dict(d["feature_schema"]),
            feature_schema_version=d.get("feature_schema_version", d.get("feature_schema", {}).get("preprocessing_version", "")),
            preprocessing_version=d.get("preprocessing_version", ""),
            configuration=d.get("configuration", {}),
            validation_metrics=d.get("validation_metrics") or d.get("metrics"),
            artifact_path=d["artifact_path"],
            status=d.get("status", "CANDIDATE"),
            created_at=d.get("created_at", ""),
            timestamp=d.get("timestamp", d.get("created_at", "")),
            approved_at=d.get("approved_at"),
            validated_at=d.get("validated_at"),
            metrics=d.get("metrics") or None,
            validation_notes=d.get("validation_notes", ""),
            integrity_hash=d.get("integrity_hash", ""),
        )


class ModelRegistry:
    """JSON-file-backed model registry with Phase 18 lifecycle.

    Layout::
        <registry_dir>/
            index.json
            active.txt              # current ACTIVE version
            history.json            # stack of previously ACTIVE versions for rollback
            artifacts/
                <version>/
                    model.pt
                    model.meta.json
    """

    def __init__(self, registry_dir: str | Path):
        self.registry_dir = Path(registry_dir)
        self.artifacts_dir = self.registry_dir / "artifacts"
        self.index_path = self.registry_dir / "index.json"
        self.history_path = self.registry_dir / "history.json"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict] = self._load_index()
        self._active: Optional[str] = self._load_active()
        self._history: List[str] = self._load_history()

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
            if version in self._index and _is_active_status(self._index[version]["status"]):
                return version
            # Legacy: check for approved as well
            if version in self._index and self._index[version]["status"] in ("ACTIVE", "approved"):
                return version
        return None

    def _save_active(self, version: Optional[str]) -> None:
        p = self.registry_dir / "active.txt"
        if version is None:
            if p.exists():
                p.unlink()
        else:
            p.write_text(version, encoding="utf-8")

    def _load_history(self) -> List[str]:
        if self.history_path.exists():
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return list(data) if isinstance(data, list) else []
            except Exception:
                return []
        return []

    def _save_history(self) -> None:
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2)

    def _file_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(
        self,
        metadata: ModelMetadata,
        artifact_source: str | Path,
        expected_input_dim: Optional[int] = None,
        copy_artifact: bool = True,
        configuration: Optional[Dict[str, Any]] = None,
        preprocessing_version: str = "",
    ) -> RegistryEntry:
        """Register a model artifact as CANDIDATE; does NOT activate it."""
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

        # Integrity hash
        try:
            integrity = self._file_hash(artifact_path) if artifact_path.exists() else ""
        except Exception:
            integrity = ""

        model_id = f"{version}-{hashlib.sha256(version.encode()).hexdigest()[:8]}"
        entry = RegistryEntry(
            model_id=model_id,
            version=version,
            algorithm=metadata.algorithm,
            training_round=metadata.training_round,
            feature_schema=metadata.feature_schema,
            feature_schema_version=metadata.feature_schema.preprocessing_version,
            preprocessing_version=preprocessing_version or metadata.feature_schema.preprocessing_version,
            configuration=configuration or {},
            validation_metrics=None,
            artifact_path=str(artifact_path),
            status="CANDIDATE",
            created_at=datetime.now(timezone.utc).isoformat(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            metrics=metadata.metrics,
            validation_notes="; ".join(notes),
            integrity_hash=integrity,
        )
        self._index[version] = entry.to_dict()
        self._save_index()
        log_event(logger, 20, f"Registered model {version} (status=CANDIDATE)", version=version)
        return entry

    # ------------------------------------------------------------------
    # Validation (Phase 18 — 6 checks)
    # ------------------------------------------------------------------
    def validate(self, version: str, expected_input_dim: Optional[int] = None,
                 expected_schema: Optional[FeatureSchema] = None,
                 expected_preprocessing: Optional[str] = None) -> tuple[bool, list[str]]:
        """Validate a CANDIDATE model before activation.

        Checks:
        1. model loads (torch.load)
        2. expected input dimensions
        3. feature schema matches
        4. preprocessing matches
        5. evaluation succeeded (validation metrics present)
        6. model integrity is valid (file exists, hash, state_dict load)
        """
        if version not in self._index:
            return False, [f"unknown version {version}"]
        data = self._index[version]
        issues: list[str] = []
        # Collect entry for reference
        try:
            entry = RegistryEntry.from_dict(data)
        except Exception as e:
            return False, [f"cannot parse entry: {e}"]

        artifact = Path(data["artifact_path"])
        # 6. integrity: file exists
        if not artifact.exists():
            issues.append(f"artifact missing: {artifact}")
        else:
            # 1. model loads
            try:
                import torch
                state = torch.load(artifact, map_location="cpu", weights_only=True)
                if not isinstance(state, dict) or len(state) == 0:
                    issues.append("model loads but state_dict is empty/invalid")
                # Additional integrity: hash
                try:
                    current_hash = self._file_hash(artifact)
                    if entry.integrity_hash and current_hash != entry.integrity_hash:
                        issues.append(f"integrity hash mismatch: expected {entry.integrity_hash[:8]} got {current_hash[:8]}")
                except Exception:
                    pass
            except Exception as e:
                issues.append(f"model loads failed (corrupted): {e}")

        # 2. expected input dimensions
        if expected_input_dim is not None and entry.feature_schema is not None:
            # ModelMetadata input_dim vs expected
            # Retrieve from configuration or metadata
            meta_dim = data.get("configuration", {}).get("model", {}).get("input_dim") if isinstance(data.get("configuration"), dict) else None
            # Fallback to feature_schema length
            schema_dim = len(entry.feature_schema.feature_names) if entry.feature_schema else None
            actual_dim = schema_dim or meta_dim
            if actual_dim is not None and actual_dim != expected_input_dim:
                issues.append(f"input_dim mismatch: expected {expected_input_dim}, got {actual_dim}")
            # Also check any prior validation_notes
        if data.get("validation_notes"):
            # validation_notes from registration (e.g., input_dim mismatch) is a hard failure
            issues.append(data["validation_notes"])

        # 3. feature schema matches
        if expected_schema is not None:
            if entry.feature_schema.preprocessing_version != expected_schema.preprocessing_version:
                issues.append(f"feature schema version mismatch: expected {expected_schema.preprocessing_version}, got {entry.feature_schema.preprocessing_version}")
            if entry.feature_schema.feature_names != expected_schema.feature_names:
                issues.append("feature schema names mismatch (incompatible schema)")

        # 4. preprocessing matches
        if expected_preprocessing is not None and entry.preprocessing_version != expected_preprocessing:
            issues.append(f"preprocessing version mismatch: expected {expected_preprocessing}, got {entry.preprocessing_version}")

        # 5. evaluation succeeded
        metrics = data.get("validation_metrics") or data.get("metrics") or {}
        # metrics must exist and contain at least one of f1/accuracy and not be empty
        if not metrics:
            issues.append("evaluation succeeded check failed: no validation metrics")
        else:
            # At least one core metric present
            if not any(k in metrics for k in ("f1", "accuracy", "precision", "recall", "roc_auc")):
                issues.append(f"evaluation metrics missing core keys: {metrics}")

        # Existing status check: cannot validate if already REJECTED/ARCHIVED
        if _canonical_status(data.get("status", "")) in ("REJECTED", "ARCHIVED"):
            issues.append(f"model already {data.get('status')} — cannot validate")

        is_valid = len(issues) == 0
        return is_valid, issues

    def mark_validated(self, version: str, validation_metrics: Optional[Dict[str, float]] = None) -> RegistryEntry:
        """Mark CANDIDATE as VALIDATED after successful validation."""
        if version not in self._index:
            raise KeyError(f"Model not registered: {version}")
        valid, issues = self.validate(version)
        if not valid:
            raise ValueError(f"Cannot mark VALIDATED — validation failed for {version}: {'; '.join(issues)}")
        self._index[version]["status"] = "VALIDATED"
        self._index[version]["validated_at"] = datetime.now(timezone.utc).isoformat()
        if validation_metrics:
            self._index[version]["validation_metrics"] = validation_metrics
            self._index[version]["metrics"] = validation_metrics
        self._save_index()
        log_event(logger, 20, f"Validated model {version} (status=VALIDATED)", version=version)
        return self.get(version)

    # ------------------------------------------------------------------
    # Activation — only VALIDATED becomes ACTIVE
    # ------------------------------------------------------------------
    def activate(self, version: str, validation_metrics: Optional[Dict[str, float]] = None) -> RegistryEntry:
        """Activate a VALIDATED model as ACTIVE (only validated models become active)."""
        if version not in self._index:
            raise KeyError(f"Model not registered: {version}")
        status = self._index[version]["status"]
        canon = _canonical_status(status)
        # Allow CANDIDATE that passes validation to be promoted directly (convenience)
        if canon == "CANDIDATE":
            valid, issues = self.validate(version)
            if not valid:
                raise ValueError(f"Cannot activate CANDIDATE {version} — validation failed: {'; '.join(issues)}")
            # Auto-promote to VALIDATED first
            self._index[version]["status"] = "VALIDATED"
            self._index[version]["validated_at"] = datetime.now(timezone.utc).isoformat()
        elif canon != "VALIDATED":
            raise ValueError(f"Only VALIDATED models can be activated, got {status} for {version}")

        # Push current active to history before replacing
        old_active = self._active
        if old_active and old_active != version:
            # Previous ACTIVE becomes ARCHIVED (keep one previous trusted for rollback)
            if self._index[old_active]["status"] in ("ACTIVE", "approved"):
                self._index[old_active]["status"] = "ARCHIVED"
            self._history.append(old_active)
            # Keep at most 5 previous
            self._history = self._history[-5:]
            self._save_history()

        self._index[version]["status"] = "ACTIVE"
        self._index[version]["approved_at"] = datetime.now(timezone.utc).isoformat()
        if validation_metrics:
            self._index[version]["validation_metrics"] = validation_metrics
            self._index[version]["metrics"] = validation_metrics
        self._save_index()
        self._active = version
        self._save_active(version)
        log_event(logger, 20, f"Activated model {version} (status=ACTIVE)", version=version)
        return self.get(version)

    # Legacy approve() — maps to VALIDATED→ACTIVE for backward compatibility
    def approve(self, version: str, metrics: Optional[dict[str, float]] = None) -> RegistryEntry:
        """Legacy approve (pending→approved). Tries VALIDATED→ACTIVE; falls back to CANDIDATE validation."""
        if version not in self._index:
            raise KeyError(f"Model not registered: {version}")
        status = self._index[version]["status"]
        canon = _canonical_status(status)
        # If already VALIDATED, just activate; if CANDIDATE, validate then activate
        if canon == "CANDIDATE":
            # Try to validate first
            valid, issues = self.validate(version)
            if not valid:
                raise ValueError(f"Cannot approve invalid model {version}: {'; '.join(issues)}")
            self._index[version]["status"] = "VALIDATED"
            self._index[version]["validated_at"] = datetime.now(timezone.utc).isoformat()
        elif canon == "VALIDATED":
            pass  # proceed to activation
        elif canon == "ACTIVE":
            return self.get(version)
        elif canon in ("REJECTED", "ARCHIVED"):
            raise ValueError(f"Cannot approve {canon} model {version}")

        return self.activate(version, validation_metrics=metrics)

    def reject(self, version: str, reason: str = "") -> RegistryEntry:
        """Reject a model (incompatible schema, corrupt weights, validation failure)."""
        if version not in self._index:
            raise KeyError(f"Model not registered: {version}")
        self._index[version]["status"] = "REJECTED"
        self._index[version]["rejected_at"] = datetime.now(timezone.utc).isoformat()
        if reason:
            self._index[version]["validation_notes"] = reason
        self._save_index()
        # If the rejected version was ACTIVE, rollback to previous trusted
        if self._active == version:
            self.rollback(reason=f"rejected active {version}: {reason}")
        return self.get(version)

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------
    def rollback(self, to_version: Optional[str] = None, reason: str = "") -> Optional[RegistryEntry]:
        """Rollback to previous trusted model.

        If to_version is None, rolls back to the most recent ARCHIVED/previous ACTIVE.
        Maintains previous trusted model(s) for recovery.
        """
        if to_version is not None:
            if to_version not in self._index:
                raise KeyError(f"Cannot rollback — unknown version {to_version}")
            # Must be a previously trusted version (ARCHIVED or ACTIVE)
            if _canonical_status(self._index[to_version]["status"]) not in ("ARCHIVED", "ACTIVE"):
                raise ValueError(f"Cannot rollback to non-trusted {to_version} status={self._index[to_version]['status']}")
            # Archive current active if any
            if self._active and self._active != to_version:
                if self._index[self._active]["status"] == "ACTIVE":
                    self._index[self._active]["status"] = "ARCHIVED"
                self._history.append(self._active)
            self._index[to_version]["status"] = "ACTIVE"
            self._active = to_version
            self._save_active(to_version)
            self._save_index()
            self._save_history()
            log_event(logger, 20, f"Rolled back to {to_version} (reason: {reason})", version=to_version)
            return self.get(to_version)

        # Auto-rollback to most recent history
        while self._history:
            candidate = self._history.pop()
            if candidate in self._index and _canonical_status(self._index[candidate]["status"]) == "ARCHIVED":
                self._index[candidate]["status"] = "ACTIVE"
                # Archive current active if still ACTIVE
                if self._active and self._active != candidate and self._active in self._index:
                    if _canonical_status(self._index[self._active]["status"]) == "ACTIVE":
                        self._index[self._active]["status"] = "ARCHIVED"
                self._active = candidate
                self._save_active(candidate)
                self._save_index()
                self._save_history()
                log_event(logger, 20, f"Auto-rollback to {candidate} (reason: {reason})", version=candidate)
                return self.get(candidate)
        # No history — try to find any ARCHIVED
        for ver, data in self._index.items():
            if _canonical_status(data["status"]) == "ARCHIVED":
                self._index[ver]["status"] = "ACTIVE"
                self._active = ver
                self._save_active(ver)
                self._save_index()
                log_event(logger, 20, f"Rollback to archived {ver} (reason: {reason})", version=ver)
                return self.get(ver)
        logger.warning("Rollback requested but no trusted previous model found (reason: %s)", reason)
        return None

    def get_active(self) -> Optional[RegistryEntry]:
        """Return the currently ACTIVE model, if any."""
        if self._active is None:
            # Try to find ACTIVE via index scan (handles legacy)
            for ver, data in self._index.items():
                if _is_active_status(data.get("status", "")):
                    self._active = ver
                    return RegistryEntry.from_dict(data)
            return None
        # Verify still ACTIVE
        data = self._index.get(self._active)
        if data and _is_active_status(data.get("status", "")):
            return self.get(self._active)
        # Stale active pointer — clear
        self._active = None
        self._save_active(None)
        return None

    def get(self, version: str) -> RegistryEntry:
        if version not in self._index:
            raise KeyError(f"Model not registered: {version}")
        return RegistryEntry.from_dict(self._index[version])

    def list_all(self) -> list[RegistryEntry]:
        """All registered models, newest first."""
        entries = [RegistryEntry.from_dict(d) for d in self._index.values()]
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries

    # Backward compat: old tests call list_models
    def list_models(self) -> list[RegistryEntry]:
        return self.list_all()

    def status_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for data in self._index.values():
            canon = _canonical_status(data["status"])
            summary[canon] = summary.get(canon, 0) + 1
        return summary

    def history(self) -> List[str]:
        return list(self._history)


def create_registry(registry_dir: str | Path) -> ModelRegistry:
    """Factory helper."""
    return ModelRegistry(registry_dir)
