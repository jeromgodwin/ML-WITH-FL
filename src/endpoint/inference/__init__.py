"""Local inference service for malware detection.

Loads the approved local model, runs inference on extracted features,
returns probability scores and metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from fedshield.config import ExperimentConfig, InferenceConfig
from fedshield.logging_setup import get_logger
from src.endpoint.feature_extraction import FeatureExtractor, FeatureVector, load_feature_schema
from src.endpoint.file_analysis import PEInfo, analyze_pe_file
from src.interfaces import DetectionRecord, FeatureSchema, ModelMetadata

logger = get_logger(__name__)


class MLP(nn.Module):
    """Small MLP matching the training architecture."""

    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...], dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # Binary classification logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class InferenceResult:
    """Result of a single inference call."""

    malware_probability: float
    benign_probability: float
    risk_score: int  # 0-100
    verdict: str  # BENIGN | MALWARE | UNCERTAIN | UNSUPPORTED | ERROR
    model_metadata: ModelMetadata
    feature_vector: FeatureVector
    inference_time_ms: float


class LocalInferenceService:
    """Local inference service for endpoint malware detection."""

    def __init__(self, config: InferenceConfig, model_config: dict[str, Any] | None = None):
        self.config = config
        self.model_config = model_config or {}
        self.device = torch.device(config.device)
        self.model: Optional[nn.Module] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.metadata: Optional[ModelMetadata] = None
        self._loaded = False

    def load(self) -> bool:
        """Load model and feature schema. Returns True on success."""
        model_path = Path(self.config.model_path)
        schema_path = Path(self.config.feature_schema_path)

        if not model_path.exists():
            logger.error("Model file not found: %s", model_path)
            return False

        if not schema_path.exists():
            logger.error("Feature schema not found: %s", schema_path)
            return False

        # Load feature schema
        try:
            schema = load_feature_schema(schema_path)
            self.feature_extractor = FeatureExtractor(schema)
        except Exception as e:
            logger.exception("Failed to load feature schema: %s", e)
            return False

        # Load model metadata (stored alongside model)
        metadata_path = model_path.with_suffix(".meta.json")
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta_dict = json.load(f)
                self.metadata = ModelMetadata.from_dict(meta_dict)
            except Exception as e:
                logger.warning("Failed to load model metadata: %s", e)

        # Build model architecture from metadata or config
        input_dim = self.metadata.input_dim if self.metadata else self.model_config.get("input_dim", 2381)
        hidden_layers = self.model_config.get("hidden_layers", (256, 128))
        dropout = self.model_config.get("dropout", 0.0)

        self.model = MLP(input_dim, hidden_layers, dropout).to(self.device)

        # Load weights
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception as e:
            logger.exception("Failed to load model weights: %s", e)
            return False

        self._loaded = True
        logger.info("Local inference service loaded: %s (v%s)", model_path, self.metadata.version if self.metadata else "unknown")
        return True

    def predict(self, path: Path) -> InferenceResult:
        """Run inference on a PE file's extracted info (static, validated)."""
        if not self._loaded:
            raise RuntimeError("Inference service not loaded. Call load() first.")

        start_time = __import__("time").time()
        path = Path(path)

        pe_info = analyze_pe_file(path)
        fv = self.feature_extractor.extract(path)

        # Run inference
        with torch.no_grad():
            x = torch.from_numpy(fv.features).unsqueeze(0).to(self.device)
            logit = self.model(x).squeeze().item()
            malware_prob = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
            benign_prob = 1.0 - malware_prob

        # Risk score (0-100)
        risk_score = int(round(malware_prob * 100))

        # Verdict based on uncertainty threshold
        if not fv.extraction_success:
            verdict = "UNSUPPORTED"
        elif malware_prob >= self.config.uncertainty_threshold:
            verdict = "MALWARE"
        elif malware_prob <= (1.0 - self.config.uncertainty_threshold):
            verdict = "BENIGN"
        else:
            verdict = "UNCERTAIN"

        inference_time_ms = (__import__("time").time() - start_time) * 1000

        return InferenceResult(
            malware_probability=malware_prob,
            benign_probability=benign_prob,
            risk_score=risk_score,
            verdict=verdict,
            model_metadata=self.metadata or ModelMetadata(
                version="unknown",
                algorithm="unknown",
                training_round=None,
                feature_schema=FeatureSchema(
                    feature_names=tuple(self.feature_extractor.schema["names"]),
                    feature_types=tuple(self.feature_extractor.schema["feature_types"]),
                    preprocessing_version=str(self.feature_extractor.schema.get("preprocessing_version") or "ember_v2_std"),
                    created_at=datetime.now().isoformat(),
                    model_version="unknown",
                ),
                metrics={},
                created_at=datetime.now().isoformat(),
                num_parameters=sum(p.numel() for p in self.model.parameters()),
                input_dim=input_dim,
            ),
            feature_vector=fv,
            inference_time_ms=inference_time_ms,
        )

    def create_detection_record(
        self,
        filepath: Path,
        pe_info: PEInfo,
        file_type: str,
        result: InferenceResult,
        action: str,
    ) -> DetectionRecord:
        """Create a DetectionRecord from inference result."""
        risk_score = int(round(result.malware_probability * 100))
        risk_level = "HIGH" if risk_score >= 80 else "MEDIUM" if risk_score >= 40 else "LOW"
        return DetectionRecord(
            detection_id=__import__("uuid").uuid4().hex[:16],
            timestamp=datetime.now().isoformat(),
            filename=filepath.name,
            filepath=str(filepath),
            sha256=pe_info.sha256,
            file_type=file_type,
            model_version=result.model_metadata.version,
            malware_probability=result.malware_probability,
            benign_probability=result.benign_probability,
            risk_score=risk_score,
            risk_level=risk_level,
            verdict=risk_level,
            action=action,
            model_algorithm=result.model_metadata.algorithm,
            analysis_duration_ms=result.inference_time_ms,
        )


def create_inference_service_from_config(config: ExperimentConfig) -> LocalInferenceService:
    """Factory to create LocalInferenceService from ExperimentConfig."""
    model_config = {
        "input_dim": config.model.input_dim,
        "hidden_layers": config.model.hidden_layers,
        "dropout": config.model.dropout,
    }
    return LocalInferenceService(config.endpoint.inference, model_config)
