"""Detection quality analysis — FP/FN, uncertain (Enhancement 8)."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


def categorize_detections(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, uncertainty_band: float = 0.1) -> Dict[str, np.ndarray]:
    """Categorize into TP/TN/FP/FN and uncertain (within threshold ± band)."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = (y_true == 1) & (y_pred == 1)
    tn = (y_true == 0) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    fn = (y_true == 1) & (y_pred == 0)
    uncertain = np.abs(y_prob - threshold) < uncertainty_band
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "uncertain": uncertain}


def analyze_quality(y_true: np.ndarray, y_prob: np.ndarray, feature_matrix: np.ndarray = None, feature_names: List[str] = None) -> Dict[str, Any]:
    """Analyze false positives/negatives: common features, confidence, etc."""
    cats = categorize_detections(y_true, y_prob)
    result = {}
    for cat, mask in cats.items():
        if np.sum(mask) == 0:
            result[cat] = {"count": 0, "mean_prob": None, "common_features": []}
            continue
        probs = y_prob[mask]
        entry = {"count": int(np.sum(mask)), "mean_prob": float(np.mean(probs)), "mean_confidence": float(np.mean(np.abs(probs - 0.5) * 2))}
        if feature_matrix is not None and feature_names is not None and np.sum(mask) > 0:
            # Common feature characteristics: mean feature value for this category
            mean_feats = np.mean(feature_matrix[mask], axis=0)
            top_idx = np.argsort(np.abs(mean_feats))[::-1][:3]
            entry["common_features"] = [{"feature": feature_names[i], "mean_value": float(mean_feats[i])} for i in top_idx]
        result[cat] = entry
    return result
