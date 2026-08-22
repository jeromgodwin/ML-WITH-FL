"""Adaptive resource budgeting — PERFORMANCE/BALANCED/BATTERY_SAVER (Enhancement 17)."""

from __future__ import annotations

def get_budget(mode: str) -> dict:
    budgets = {
        "PERFORMANCE": {"max_cpu": 90, "max_rounds": 10, "local_epochs": 5},
        "BALANCED": {"max_cpu": 70, "max_rounds": 5, "local_epochs": 3},
        "BATTERY_SAVER": {"max_cpu": 50, "max_rounds": 1, "local_epochs": 1},
    }
    return budgets.get(mode, budgets["BALANCED"])
