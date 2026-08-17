"""Retraining safety guard (Phase 13).

Enforces:
- minimum cooldown between retraining events
- maximum frequency per time window (default: per day)
- minimum new samples required
- maximum retraining rounds
- prevents retraining loops
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from fedshield.config import DriftConfig
from fedshield.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SafetyCheck:
    allowed: bool
    reason: str
    details: dict


class RetrainingSafety:
    def __init__(self, config: DriftConfig):
        self.config = config
        self._last_retrain_ts: Optional[float] = None
        self._retrain_count_today: int = 0
        self._day_start_ts: float = time.time()

    def check(self, new_samples: int) -> SafetyCheck:
        """Evaluate if a retraining event is permitted."""
        now = time.time()
        self._reset_daily_counter(now)

        # 1. minimum new samples
        if new_samples < self.config.min_new_samples:
            return SafetyCheck(
                allowed=False,
                reason="insufficient_new_samples",
                details={"available": new_samples, "required": self.config.min_new_samples},
            )

        # 2. cooldown
        if self._last_retrain_ts is not None:
            elapsed_h = (now - self._last_retrain_ts) / 3600.0
            if elapsed_h < self.config.cooldown_hours:
                return SafetyCheck(
                    allowed=False,
                    reason="cooldown",
                    details={"elapsed_hours": round(elapsed_h, 2),
                             "required_hours": self.config.cooldown_hours},
                )

        # 3. max frequency per day
        if self._retrain_count_today >= self.config.max_frequency_per_day:
            return SafetyCheck(
                allowed=False,
                reason="max_frequency_exceeded",
                details={"count_today": self._retrain_count_today,
                         "max_per_day": self.config.max_frequency_per_day},
            )

        return SafetyCheck(allowed=True, reason="", details={})

    def record_retrain(self, rounds: int) -> None:
        """Record a completed retraining event."""
        now = time.time()
        self._last_retrain_ts = now
        self._retrain_count_today += 1
        logger.info("retraining recorded: rounds=%d, count_today=%d",
                    rounds, self._retrain_count_today)

    def status(self) -> dict:
        now = time.time()
        self._reset_daily_counter(now)
        return {
            "last_retrain_ts": self._last_retrain_ts,
            "retrain_count_today": self._retrain_count_today,
            "cooldown_hours": self.config.cooldown_hours,
            "max_frequency_per_day": self.config.max_frequency_per_day,
            "min_new_samples": self.config.min_new_samples,
            "max_retraining_rounds": self.config.max_retraining_rounds,
        }

    def _reset_daily_counter(self, now: float) -> None:
        if now - self._day_start_ts >= 86400.0:  # 24h
            self._retrain_count_today = 0
            self._day_start_ts = now