"""Canary model rollout — candidate → canary → monitor → broader rollout (Enhancement 15)."""

from __future__ import annotations

def should_promote(canary_f1: float, old_f1: float, threshold: float = 0.01) -> bool:
    return canary_f1 >= old_f1 - threshold
