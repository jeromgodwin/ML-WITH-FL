"""Client health and trust score — HEALTHY/MONITOR/SUSPICIOUS/RESTRICTED (Enhancement 16)."""

from __future__ import annotations

def score_client(successful_rounds: int, anomalies: int, failures: int) -> str:
    if failures > 5 or anomalies > 3:
        return "RESTRICTED"
    if anomalies > 1 or failures > 2:
        return "SUSPICIOUS"
    if anomalies > 0:
        return "MONITOR"
    return "HEALTHY"
