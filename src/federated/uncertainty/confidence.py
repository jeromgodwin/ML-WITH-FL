"""Model confidence and uncertainty — HIGH/MEDIUM/LOW (Enhancement 6)."""

from __future__ import annotations

from typing import Tuple

def confidence_category(prob: float, thresholds: Tuple[float, float] = (0.2, 0.4)) -> str:
    """Map malware probability to confidence.

    Distance from 0.5 determines confidence, not safety.
    thresholds: (low_medium, medium_high) on |p-0.5|
    """
    d = abs(prob - 0.5)
    low, high = thresholds
    if d > high:
        return "HIGH CONFIDENCE"
    if d > low:
        return "MEDIUM CONFIDENCE"
    return "LOW CONFIDENCE"

def policy_for_uncertain(prob: float, base_action: str, confidence: str) -> str:
    """Low-confidence → WARNING rather than QUARANTINE, configurable."""
    if confidence == "LOW CONFIDENCE" and base_action == "QUARANTINE":
        return "WARN"
    return base_action
