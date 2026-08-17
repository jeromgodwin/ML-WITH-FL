"""Static PE feature extraction — Phase 6.

Dataset-feature compatibility contract
--------------------------------------
The model was trained on EMBER 2018_2 v2 vectors (2381 features, canonical
order defined in src/federated/data/feature_schema.py). This module is the
REAL extraction method for detected files: lief 1.0 parse -> official raw
feature blocks -> official process_raw_features vectorization, producing a
vector equivalent to the dataset rows the model saw.

Every extracted vector is validated before inference:

- feature count == schema count == 2381
- dtype float32
- no NaN/Inf
- feature order == canonical Phase-2 schema order (schema checked at load)
- preprocessing version compatible (feature_version == 2)

Validation failures raise ExtractionError with an explicit message. Missing
or malformed values are NEVER silently replaced with constants.

Output is a standardized FeatureVector (hash, file metadata, values, schema
version, extraction timestamp). Raw feature rows are not exposed to UIs by
default (to_dict(expose_features=False)).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from fedshield.logging_setup import get_logger
from src.endpoint.ember_features import extract_feature_vector as _ember_extract
from src.endpoint.file_analysis import compute_sha256
from src.federated.data.feature_schema import (
    FEATURE_NAMES, FEATURE_VERSION, N_FEATURES,
)

logger = get_logger(__name__)


class ExtractionError(Exception):
    """Raised when a file cannot be safely/conformantly vectorized."""


@dataclass
class FeatureVector:
    """Standardized extracted feature vector for one file."""

    features: np.ndarray          # float32 (2381,)
    feature_names: List[str]
    schema_version: str
    model_version: str
    extraction_success: bool
    missing_features: List[str]
    extra_features: List[str]
    # Phase 6 metadata
    sha256: str = ""
    file_path: str = ""
    file_size: int = 0
    extracted_at: str = ""

    def to_dict(self, expose_features: bool = False) -> Dict[str, Any]:
        """JSON-safe summary; raw feature rows are private by default."""
        d: Dict[str, Any] = {
            "sha256": self.sha256,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "extracted_at": self.extracted_at,
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "n_features": int(self.features.shape[0]),
            "extraction_success": self.extraction_success,
            "missing_features": self.missing_features,
            "extra_features": self.extra_features,
        }
        if expose_features:
            d["features"] = self.features.tolist()
        return d


# ---------------------------------------------------------------------------
# schema loading / validation
# ---------------------------------------------------------------------------

def load_feature_schema(path: Path) -> Dict[str, Any]:
    """Load a feature schema JSON; supports the bundle format and the
    interfaces FeatureSchema format. Returns a normalized dict."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if "names" in raw:  # bundle format: {feature_version, n_features, names}
        return {
            "names": list(raw["names"]),
            "feature_version": int(raw.get("feature_version", -1)),
            "feature_types": ("float32",) * len(raw["names"]),
            "preprocessing_version": f"ember_v{raw.get('feature_version')}_std",
            "model_version": raw.get("model_version", ""),
        }
    if "feature_names" in raw:  # interfaces FeatureSchema format
        return {
            "names": list(raw["feature_names"]),
            "feature_version": 2 if "v2" in str(raw.get("preprocessing_version", "")) else -1,
            "feature_types": tuple(raw.get("feature_types", ())),
            "preprocessing_version": raw.get("preprocessing_version", ""),
            "model_version": raw.get("model_version", ""),
        }
    raise ExtractionError(f"unrecognized feature schema format in {path}")


def validate_schema(schema: Dict[str, Any]) -> List[str]:
    """Strict schema checks: count, types, exact order, preprocessing version.

    Returns a list of errors (empty = schema is conformant). The order check
    is exact: the schema must list the canonical Phase-2 names in canonical
    order; no reordering or renaming is accepted.
    """
    errors: List[str] = []
    names = schema.get("names") or []
    if len(names) != N_FEATURES:
        errors.append(f"schema has {len(names)} features, expected {N_FEATURES}")
    if names != FEATURE_NAMES:
        errors.append("schema feature names/order do not match the canonical Phase-2 schema")
    types = schema.get("feature_types") or ()
    if types and set(types) - {"float32", "<f4"}:
        errors.append(f"unsupported feature types: {sorted(set(types))}")
    if schema.get("feature_version") != FEATURE_VERSION:
        errors.append(
            f"incompatible preprocessing version: feature_version="
            f"{schema.get('feature_version')}, expected {FEATURE_VERSION}")
    return errors


def validate_vector(features: np.ndarray, schema: Dict[str, Any]) -> List[str]:
    """Pre-inference vector validation; returns a list of errors (empty = OK).

    No values are modified or replaced here; any violation surfaces as an
    explicit error for the caller.
    """
    errors: List[str] = []
    expected = len(schema.get("names") or [])
    if not isinstance(features, np.ndarray):
        return ["features is not a numpy array"]
    if features.ndim != 1 or features.shape[0] != expected:
        errors.append(f"feature count {features.shape} != schema count {expected}")
    if features.dtype != np.float32:
        errors.append(f"feature dtype {features.dtype}, expected float32")
    if not np.isfinite(features).all():
        errors.append("vector contains NaN/Inf values")
    return errors


# ---------------------------------------------------------------------------
# the real extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Extracts validated EMBER v2 vectors from real PE files (lief 1.0).

    ``schema`` may be a normalized schema dict, a path to a schema JSON, or
    an interfaces FeatureSchema. Schema conformity is checked eagerly at
    construction; per-file vectors are validated in ``extract``.
    """

    def __init__(self, schema: Dict[str, Any] | Path | str | Any = None):
        if schema is None:
            schema = {"names": FEATURE_NAMES, "feature_version": FEATURE_VERSION}
        if isinstance(schema, (str, Path)):
            schema = load_feature_schema(Path(schema))
        elif hasattr(schema, "feature_names"):  # interfaces FeatureSchema
            schema = {
                "names": list(schema.feature_names),
                "feature_types": tuple(schema.feature_types),
                "preprocessing_version": schema.preprocessing_version,
                "model_version": schema.model_version,
                "feature_version": 2 if "v2" in str(schema.preprocessing_version) else -1,
            }
        errors = validate_schema(schema)
        if errors:
            raise ExtractionError("feature schema rejected: " + "; ".join(errors))
        self.schema = schema
        self.expected_dim = len(schema["names"])
        logger.info("FeatureExtractor ready: %d features, order=canonical v%d",
                    self.expected_dim, schema.get("feature_version"))

    @classmethod
    def from_schema_file(cls, schema_path: Path) -> "FeatureExtractor":
        return cls(load_feature_schema(schema_path))

    def extract(self, path: Path) -> FeatureVector:
        """Extract + validate the feature vector for ``path`` (static, read-only).

        Raises ExtractionError when the file is not a parseable PE or the
        vector fails validation.
        """
        path = Path(path)
        if not path.is_file():
            raise ExtractionError(f"not a file: {path}")
        try:
            bytez = path.read_bytes()  # read-only; the file is never executed
        except OSError as exc:
            raise ExtractionError(f"cannot read {path}: {exc}") from exc
        if len(bytez) < 2 or bytez[:2] != b"MZ":
            raise ExtractionError(f"not a PE file (no MZ signature): {path}")

        vec = _ember_extract(bytez)
        if vec is None:
            raise ExtractionError(f"PE parse/vectorization failed: {path}")

        errors = validate_vector(vec, self.schema)
        if errors:
            raise ExtractionError(f"extracted vector failed validation: {'; '.join(errors)}")

        return FeatureVector(
            features=vec.astype(np.float32, copy=False),
            feature_names=list(self.schema["names"]),
            schema_version=str(self.schema.get("preprocessing_version")
                               or f"ember_v{self.schema.get('feature_version')}_std"),
            model_version=str(self.schema.get("model_version") or ""),
            extraction_success=True,
            missing_features=[],
            extra_features=[],
            sha256=compute_sha256(path),
            file_path=str(path),
            file_size=int(path.stat().st_size),
            extracted_at=datetime.now(timezone.utc).isoformat(),
        )


# ---------------------------------------------------------------------------
# training-side schema helpers (kept for compatibility with model export)
# ---------------------------------------------------------------------------

def create_feature_schema_from_training(
    feature_names: List[str],
    preprocessing_version: str,
    model_version: str,
) -> Dict[str, Any]:
    """Create a normalized schema dict from training-pipeline output."""
    return {
        "names": list(feature_names),
        "feature_version": FEATURE_VERSION,
        "feature_types": ("float32",) * len(feature_names),
        "preprocessing_version": preprocessing_version,
        "model_version": model_version,
    }


def save_feature_schema(schema: Dict[str, Any], path: Path) -> None:
    """Save a normalized schema dict to JSON (bundle format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "feature_version": schema.get("feature_version"),
            "n_features": len(schema.get("names") or []),
            "names": schema.get("names") or [],
        }, f, indent=2)