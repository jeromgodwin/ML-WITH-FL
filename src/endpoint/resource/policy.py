"""Resource policy: config-driven permit/pause/cancel decisions (Phase 12).

The policy is a pure function of (config, snapshot, elapsed training time):
no state, no timing, so it is trivially testable. All thresholds come from
``ResourceConfig`` (configured in configs/default.yaml); constraints whose
threshold is unset OR whose metric is unavailable are skipped, so a policy
never fails on a machine that cannot measure a signal.

Decision actions:
    permit  training may run
    pause   training must defer (re-check later)
    cancel  training must stop (max duration exceeded)

Reasons: high_cpu, low_battery, on_battery, user_active, low_memory,
max_duration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fedshield.config import ResourceConfig


@dataclass(frozen=True)
class PolicyDecision:
    action: str  # permit | pause | cancel
    reason: str = ""
    # constraints skipped because the metric was unavailable (informational)
    unavailable: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"action": self.action, "reason": self.reason,
                "unavailable": list(self.unavailable)}


class ResourcePolicy:
    def __init__(self, config: ResourceConfig):
        self.config = config

    # ------------------------------------------------------------------
    def decide(self, snapshot: dict, training_elapsed_sec: float) -> PolicyDecision:
        """Evaluate the policy against one resource snapshot.

        ``training_elapsed_sec`` is the time since the current fit started;
        only the max-duration constraint may CANCEL (all others pause and
        re-check later).
        """
        if not self.config.enabled:
            return PolicyDecision("permit")

        unavailable = [name for name, flag in (
            ("cpu", snapshot.get("cpu_available")),
            ("ram", snapshot.get("ram_available")),
            ("battery", snapshot.get("battery_available")),
            ("activity", snapshot.get("activity_available")),
        ) if not flag]

        # 1. CPU utilization
        if (self.config.max_cpu_percent is not None
                and snapshot.get("cpu_available")):
            if snapshot["cpu_percent"] > self.config.max_cpu_percent:
                return PolicyDecision("pause", "high_cpu", tuple(unavailable))

        # 2. RAM
        if (self.config.min_free_memory_mb is not None
                and snapshot.get("ram_available")
                and snapshot.get("ram_free_mb") is not None):
            if snapshot["ram_free_mb"] < self.config.min_free_memory_mb:
                return PolicyDecision("pause", "low_memory", tuple(unavailable))

        # 3. Battery / AC
        if snapshot.get("battery_available"):
            if (self.config.require_ac_power
                    and snapshot.get("ac_powered") is False):
                return PolicyDecision("pause", "on_battery", tuple(unavailable))
            if (self.config.min_battery_percent is not None
                    and snapshot.get("battery_percent") is not None
                    and snapshot["battery_percent"] < self.config.min_battery_percent):
                return PolicyDecision("pause", "low_battery", tuple(unavailable))

        # 4. User activity
        if (self.config.idle_only and snapshot.get("activity_available")
                and snapshot.get("idle_seconds") is not None):
            threshold = self.config.idle_min_seconds
            if threshold is not None and snapshot["idle_seconds"] < threshold:
                return PolicyDecision("pause", "user_active", tuple(unavailable))

        # 5. Max training duration (hard stop, not a pause)
        if (self.config.max_training_duration_sec is not None
                and training_elapsed_sec >= self.config.max_training_duration_sec):
            return PolicyDecision("cancel", "max_duration", tuple(unavailable))

        return PolicyDecision("permit", "", tuple(unavailable))