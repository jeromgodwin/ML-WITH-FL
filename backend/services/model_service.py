"""Model service (Phase 20 §7).

Expose model versions, active model, candidate models, validation status, rollback info.
Delegates to ModelRegistry — no direct file access in handlers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.federated.model_registry import ModelRegistry


class ModelService:
    def __init__(self, registry: Optional[ModelRegistry] = None, registry_dir: Path | str = "data/server_registry"):
        self.registry = registry or ModelRegistry(registry_dir)

    def list_versions(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.registry.list_all()]

    def get_active(self) -> Optional[Dict[str, Any]]:
        a = self.registry.get_active()
        return a.to_dict() if a else None

    def list_candidates(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.registry.list_all() if e.status in ("CANDIDATE", "pending")]

    def get_validation(self, version: str) -> Optional[Dict[str, Any]]:
        try:
            e = self.registry.get(version)
            valid, issues = self.registry.validate(version)
            return {"version": version, "status": e.status, "valid": valid, "issues": issues, "validation_metrics": e.validation_metrics}
        except KeyError:
            return None

    def rollback(self, to_version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        try:
            entry = self.registry.rollback(to_version=to_version)
            return entry.to_dict() if entry else None
        except Exception as e:
            return {"error": str(e)}

    def history(self) -> List[str]:
        return self.registry.history()
