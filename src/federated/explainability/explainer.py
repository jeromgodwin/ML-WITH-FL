"""Feature importance for local malware classifier (Enhancement 4).

For every detection, produce understandable explanation based on actual model inputs.
Uses model-specific importance: first linear layer weights * feature value.
Clearly distinguishes model explanation from malware certainty.
No fabricated reasons — uses actual feature names from schema.
Measures explanation latency.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from src.federated.data.feature_schema import FEATURE_NAMES


class FeatureExplainer:
    """Explains a prediction via first-layer weight * feature contribution."""

    def __init__(self, model: torch.nn.Module, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names or list(FEATURE_NAMES[:2381])
        # Extract first linear weights: shape (hidden, input_dim)
        first_linear = None
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                first_linear = module
                break
        if first_linear is None:
            raise ValueError("model has no Linear layer")
        # Mean absolute weight per input feature (across hidden units)
        with torch.no_grad():
            w = first_linear.weight.detach().cpu().numpy()  # (hidden, input)
            self.feature_importance = np.mean(np.abs(w), axis=0)  # (input_dim,)

    def explain(self, feature_vector: np.ndarray, top_k: int = 3) -> Tuple[List[Dict[str, Any]], float]:
        """Return top-k contributing features and latency ms.

        Contribution = feature_value * importance (actual model input * weight).
        """
        t0 = time.perf_counter()
        feature_vector = np.asarray(feature_vector, dtype=np.float32).ravel()
        if feature_vector.shape[0] != self.feature_importance.shape[0]:
            # Truncate or pad
            n = min(feature_vector.shape[0], self.feature_importance.shape[0])
            feature_vector = feature_vector[:n]
            importance = self.feature_importance[:n]
            names = self.feature_names[:n]
        else:
            importance = self.feature_importance
            names = self.feature_names

        contributions = feature_vector * importance
        # Top-k by absolute contribution
        idx = np.argsort(np.abs(contributions))[::-1][:top_k]
        result = []
        for i in idx:
            result.append({
                "feature": names[i] if i < len(names) else f"f{i}",
                "feature_index": int(i),
                "feature_value": float(feature_vector[i]),
                "importance": float(importance[i]),
                "contribution": float(contributions[i]),
            })
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return result, latency_ms


def explain_prediction(model: torch.nn.Module, feature_vector: np.ndarray, feature_names: List[str] = None, top_k: int = 3) -> Dict[str, Any]:
    """Convenience: explain a single prediction with model explanation vs certainty distinction."""
    explainer = FeatureExplainer(model, feature_names)
    top_features, latency = explainer.explain(feature_vector, top_k=top_k)
    # Get probability for certainty distinction
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(feature_vector, dtype=np.float32).reshape(1, -1))).numpy().ravel()[0]
        prob = float(1.0 / (1.0 + np.exp(-logits)))
    return {
        "model_explanation": {
            "top_features": top_features,
            "latency_ms": latency,
            "note": "Top contributing signals based on actual feature values * first-layer weights — not proof of maliciousness",
        },
        "malware_certainty": {
            "malware_probability": prob,
            "note": "Model's malware probability — distinct from explanation",
        },
    }
