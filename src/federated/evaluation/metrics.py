"""Binary classification evaluation metrics and chunked model evaluation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Compute accuracy/precision/recall/F1/ROC-AUC/confusion matrix.

    y_true: 0/1 labels; y_prob: predicted probabilities in [0, 1].
    ROC-AUC is reported when both classes are present, else None.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = None
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_prob))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)),
    }


@torch.no_grad()
def predict_proba_chunked(
    model: torch.nn.Module,
    X: np.ndarray,
    chunk: int = 20000,
    device: str = "cpu",
    scale_inv: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Predict probabilities for X in chunks (bounds RAM for large datasets).

    scale_inv: optional per-feature multiplier applied to each chunk before
    inference (e.g., 1/scale from a StandardScaler); lets callers stream
    chunks straight from a memmap without materializing the whole array.
    """
    model = model.to(device)
    model.eval()
    out = np.empty(X.shape[0], dtype=np.float32)
    for start in range(0, X.shape[0], chunk):
        xb = np.asarray(X[start:start + chunk], dtype=np.float32)
        if scale_inv is not None:
            xb = xb * scale_inv
        xb_t = torch.from_numpy(xb)
        logits = model(xb_t.to(device))
        out[start:start + chunk] = torch.sigmoid(logits).cpu().numpy().ravel()
    return out


def evaluate_model(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    chunk: int = 20000,
    device: str = "cpu",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Full evaluation: metrics on (X, y) with chunked inference."""
    y_prob = predict_proba_chunked(model, X, chunk=chunk, device=device)
    return compute_metrics(y, y_prob, threshold=threshold)