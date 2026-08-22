"""Advanced drift monitoring — feature/class/client/global/temporal, LOW/MEDIUM/HIGH (Enhancement 14)."""

from __future__ import annotations

from typing import Dict, Any

def drift_severity(psi: float) -> str:
    if psi >= 0.3:
        return "HIGH"
    if psi >= 0.15:
        return "MEDIUM"
    return "LOW"
