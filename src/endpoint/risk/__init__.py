"""Risk engine: maps malware probability to a policy level and decision.

The thresholds here are CONFIGURATION, not universal truth. They encode one
deployment's risk tolerance and must be tuned per environment. No claim is
made that these specific cut-offs generalize.
"""

from __future__ import annotations

from typing import Tuple

from fedshield.config import RiskConfig
from fedshield.logging_setup import get_logger

logger = get_logger(__name__)

# risk_level bands on the 0-100 risk score
LEVEL_BANDS = (("LOW", 0, 39), ("MEDIUM", 40, 79), ("HIGH", 80, 100))


class RiskEngine:
    """Maps a malware probability to risk_score, risk_level, and action."""

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        allow_max, warn_max = self.config.thresholds
        if not (0.0 <= allow_max <= warn_max <= 1.0):
            raise ValueError(
                f"invalid risk thresholds {self.config.thresholds}: "
                f"expected 0 <= allow_max <= warn_max <= 1")
        self.allow_max = allow_max
        self.warn_max = warn_max
        logger.debug("risk engine: allow<%.2f warn<%.2f -> %s",
                     allow_max, warn_max, self.config.actions)

    def action(self, p: float) -> str:
        """Decision for a malware probability: ALLOW | WARN | QUARANTINE."""
        allow, warn, quarantine = self.config.actions
        if p < self.allow_max:
            return allow
        if p < self.warn_max:
            return warn
        return quarantine

    def risk_score(self, p: float) -> int:
        """Continuous 0-100 score (linear in probability)."""
        return int(round(min(max(p, 0.0), 1.0) * 100))

    def level(self, p: float) -> str:
        """Policy level for a probability: LOW | MEDIUM | HIGH."""
        score = self.risk_score(p)
        for name, lo, hi in LEVEL_BANDS:
            if lo <= score <= hi:
                return name
        return "LOW"

    def decide(self, p: float) -> Tuple[int, str, str, str]:
        """Full risk decision -> (risk_score, risk_level, verdict, action).

        ``verdict`` is the policy level (LOW/MEDIUM/HIGH), ``action`` is the
        configured decision (ALLOW/WARN/QUARANTINE).
        """
        p = min(max(p, 0.0), 1.0)
        return self.risk_score(p), self.level(p), self.level(p), self.action(p)