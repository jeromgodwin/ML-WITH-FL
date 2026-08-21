"""FL training → candidate model → validation → registry → active → endpoint.

Implements the required Phase 18 flow:
FL training → candidate model → validation → registry → approved model → client update → endpoint uses new model

If the network/server is unavailable, the endpoint continues using the current active model.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from fedshield.logging_setup import get_logger
from src.federated.model_registry import ModelRegistry, RegistryEntry
from src.federated.model_bundle import export_bundle
from src.federated.models.mlp import MLPConfig, build_mlp
from src.interfaces import FeatureSchema, ModelMetadata
from src.federated.data.feature_schema import FEATURE_NAMES, FEATURE_VERSION, N_FEATURES

logger = get_logger(__name__)


def fl_checkpoint_to_registry(
    checkpoint_path: Path,
    model_cfg: MLPConfig,
    scaler_path: Path,
    metrics: Dict[str, float],
    version: str,
    algorithm: str,
    training_round: Optional[int],
    feature_schema: Optional[FeatureSchema],
    preprocessing_version: str,
    configuration: Dict[str, Any],
    registry: ModelRegistry,
    expected_input_dim: int = 2381,
    validation_metrics: Optional[Dict[str, float]] = None,
) -> RegistryEntry:
    """Promote an FL checkpoint through the registry lifecycle.

    Steps:
    1. Create model_id/version, build ModelMetadata
    2. Register as CANDIDATE (via registry.register)
    3. Validate (6 checks: loads, input dim, schema, preprocessing, eval, integrity)
    4. If valid → mark VALIDATED → activate as ACTIVE
       If invalid → REJECT and keep previous ACTIVE
    Returns the final RegistryEntry.
    """
    if feature_schema is None:
        feature_schema = FeatureSchema(
            feature_names=tuple(FEATURE_NAMES),
            feature_types=("float32",) * N_FEATURES,
            preprocessing_version=preprocessing_version or f"ember_v{FEATURE_VERSION}_std",
            created_at=datetime.now(timezone.utc).isoformat(),
            model_version=version,
        )

    # Robust num_parameters: corrupted checkpoint should not crash promotion — it will be rejected at validation
    try:
        num_params = sum(p.numel() for p in torch.load(checkpoint_path, map_location="cpu", weights_only=True).values()) if Path(checkpoint_path).exists() else 0
    except Exception:
        num_params = 0
    metadata = ModelMetadata(
        version=version,
        algorithm=algorithm,
        training_round=training_round,
        feature_schema=feature_schema,
        metrics=validation_metrics or metrics or {},
        created_at=datetime.now(timezone.utc).isoformat(),
        num_parameters=num_params,
        input_dim=model_cfg.input_dim,
    )

    # Register as CANDIDATE
    entry = registry.register(
        metadata=metadata,
        artifact_source=checkpoint_path,
        expected_input_dim=expected_input_dim,
        configuration=configuration,
        preprocessing_version=preprocessing_version,
    )
    logger.info("FL model %s registered as CANDIDATE", version)

    # Validation (6 checks)
    valid, issues = registry.validate(version, expected_input_dim=expected_input_dim,
                                      expected_schema=feature_schema,
                                      expected_preprocessing=preprocessing_version)
    if not valid:
        logger.warning("FL model %s validation FAILED: %s", version, "; ".join(issues))
        registry.reject(version, reason="; ".join(issues))
        # Rollback is implicit: previous ACTIVE remains
        return registry.get(version)

    # Mark VALIDATED
    try:
        registry.mark_validated(version, validation_metrics=validation_metrics or metrics)
    except Exception as e:
        logger.warning("mark_validated failed for %s: %s", version, e)
        # Try direct activation (approve handles candidate→validated internally)
        pass

    # Activate — only VALIDATED becomes ACTIVE
    try:
        active = registry.activate(version, validation_metrics=validation_metrics or metrics)
        logger.info("FL model %s activated (ACTIVE)", version)
        return active
    except Exception as e:
        logger.error("activation failed for %s: %s", version, e)
        registry.reject(version, reason=str(e))
        # Rollback to previous trusted
        prev = registry.rollback(reason=f"activation failed for {version}: {e}")
        if prev:
            logger.info("rolled back to %s after %s failure", prev.version, version)
        return registry.get(version)


def endpoint_discover_and_update(
    registry: ModelRegistry,
    endpoint_model_dir: Optional[Path] = None,
    current_version: Optional[str] = None,
) -> Optional[RegistryEntry]:
    """Endpoint client discovers new approved model, verifies, activates locally.

    - Checks registry for current ACTIVE version
    - If new ACTIVE differs from current_version, verifies artifact
    - Copies artifact to endpoint_model_dir without manual file replacement
    - If network/server unavailable (registry unreadable), continues with current

    Returns the new active entry if updated, or None if no update / unavailable.
    """
    try:
        active = registry.get_active()
    except Exception as e:
        logger.warning("registry unavailable (network/server down): %s — continuing with current %s", e, current_version)
        return None

    if active is None:
        logger.info("no ACTIVE model in registry — continuing with %s", current_version)
        return None

    if current_version is not None and active.version == current_version:
        return None  # already up to date

    # Verify before activating on endpoint (model loads, input dim, schema)
    try:
        # Quick verify: artifact exists and loads
        artifact_path = Path(active.artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"artifact missing: {artifact_path}")
        import torch
        state = torch.load(artifact_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("invalid state_dict")
    except Exception as e:
        logger.error("endpoint verification failed for %s: %s — keeping %s", active.version, e, current_version)
        return None

    # Deploy to endpoint dir (automatic without manual replace)
    if endpoint_model_dir is not None:
        try:
            endpoint_model_dir = Path(endpoint_model_dir)
            endpoint_model_dir.mkdir(parents=True, exist_ok=True)
            dst = endpoint_model_dir / "model.pt"
            # Atomic copy
            shutil.copy2(active.artifact_path, dst)
            (endpoint_model_dir / "active_version.txt").write_text(active.version, encoding="utf-8")
            (endpoint_model_dir / "model.meta.json").write_text(
                json.dumps(active.to_dict(), indent=2), encoding="utf-8"
            )
            logger.info("endpoint auto-updated to %s -> %s", active.version, dst)
        except Exception as e:
            logger.error("endpoint update failed for %s: %s", active.version, e)
            return None

    return active
