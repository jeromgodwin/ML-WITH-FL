"""Training controls for resource-aware FL (Phase 12).

The controller is the SINGLE coordination point between federated training
and the rest of the endpoint, and it is deliberately one-way: ONLY the FL
training thread ever touches it. Real-time protection (file monitoring,
feature extraction, inference, risk, notifications, quarantine) never
consults it and therefore can never be blocked by it.

State machine:
    idle -> started -> paused <-> started -> finished
    any   -> cancelled (terminal)

Controls: request_start, pause, resume, cancel, finish.
Training: wait_until_allowed() blocks until the resource policy permits
(maybe after re-sampling), returns False only on cancel.

Everything is recorded: pause count, total paused/wait time, last pause
reason, and an append-only event log (JSON-serializable for the dashboard).
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import Optional

from fedshield.config import ResourceConfig
from fedshield.logging_setup import get_logger

from src.endpoint.resource.monitor import ResourceMonitor
from src.endpoint.resource.policy import PolicyDecision, ResourcePolicy

logger = get_logger(__name__)

STATE_IDLE = "idle"
STATE_STARTED = "started"
STATE_PAUSED = "paused"
STATE_CANCELLED = "cancelled"
STATE_FINISHED = "finished"

TERMINAL_STATES = (STATE_CANCELLED, STATE_FINISHED)


class TrainingController:
    def __init__(
        self,
        policy: ResourcePolicy,
        monitor: ResourceMonitor,
        check_interval_sec: Optional[float] = None,
    ):
        self.policy = policy
        self.monitor = monitor
        self.check_interval_sec = check_interval_sec or policy.config.check_interval_sec
        self._lock = threading.Lock()
        self._state = STATE_IDLE
        self._started_at: Optional[float] = None
        self._paused_at: Optional[float] = None
        self._paused_total_sec = 0.0
        self._wait_total_sec = 0.0
        self._pauses = 0
        self._last_pause_reason = ""
        self._last_decision: Optional[PolicyDecision] = None
        self._events: list[dict] = []

    # ------------------------------------------------------------------
    # control surface (start / pause / resume / cancel)
    # ------------------------------------------------------------------
    def request_start(self) -> bool:
        """Enter 'started'. Returns False if already started/cancelled."""
        with self._lock:
            if self._state in (STATE_STARTED, STATE_PAUSED):
                return False
            if self._state == STATE_CANCELLED:
                return False
            self._state = STATE_STARTED
            self._started_at = time.time()
            self._record("start")
        return True

    def pause(self, reason: str = "manual") -> bool:
        """Defer training (idempotent). Real-time protection is unaffected."""
        with self._lock:
            if self._state != STATE_STARTED:
                return False
            self._state = STATE_PAUSED
            self._paused_at = time.time()
            self._pauses += 1
            self._last_pause_reason = reason
            self._record("pause", reason=reason)
        logger.info("FL training paused (%s)", reason)
        return True

    def resume(self) -> bool:
        """Resume after a pause (no-op unless paused)."""
        with self._lock:
            if self._state != STATE_PAUSED:
                return False
            self._paused_total_sec += time.time() - (self._paused_at or time.time())
            self._paused_at = None
            self._state = STATE_STARTED
            self._record("resume")
        logger.info("FL training resumed")
        return True

    def cancel(self, reason: str = "manual") -> bool:
        """Abort training permanently (terminal; idempotent)."""
        with self._lock:
            if self._state in TERMINAL_STATES:
                return False
            if self._state == STATE_PAUSED and self._paused_at is not None:
                self._paused_total_sec += time.time() - self._paused_at
            self._state = STATE_CANCELLED
            self._record("cancel", reason=reason)
        logger.info("FL training cancelled (%s)", reason)
        return True

    def finish(self) -> None:
        """Mark training complete; record final totals (idempotent)."""
        with self._lock:
            if self._state in TERMINAL_STATES:
                return
            if self._state == STATE_PAUSED and self._paused_at is not None:
                self._paused_total_sec += time.time() - self._paused_at
                self._paused_at = None
            self._state = STATE_FINISHED
            self._record("finish")

    # ------------------------------------------------------------------
    # training-side gate
    # ------------------------------------------------------------------
    def gate(self) -> PolicyDecision:
        """One policy evaluation of the CURRENT snapshot (no waiting).

        The decision does not mutate the controller's state machine — it is
        recorded as the latest decision and returned; ``wait_until_allowed``
        is the only place that acts on decisions.
        """
        elapsed = 0.0
        with self._lock:
            if self._started_at is not None:
                elapsed = time.time() - self._started_at
        decision = self.policy.decide(self.monitor.snapshot(), elapsed)
        with self._lock:
            self._last_decision = decision
        return decision

    def wait_until_allowed(self, timeout: Optional[float] = None) -> bool:
        """Block until training is permitted; False only on CANCEL.

        Re-samples the resource snapshot every ``check_interval_sec`` while
        the policy defers training (or a manual pause is in effect) and
        returns True the moment the policy permits. Returns False when
        cancelled — by the max-duration policy or by a manual cancel.
        ``timeout`` caps the total wait (None = wait indefinitely). Each
        continuous deferral block is recorded as a pause.
        """
        wait_start = time.time()
        deferred_at: Optional[float] = None
        while True:
            with self._lock:
                state = self._state
            if state == STATE_CANCELLED:
                return False
            if state == STATE_PAUSED:
                # manual pause: keep waiting (only a resume or cancel exits)
                if deferred_at is None:
                    deferred_at = time.time()
                if timeout is not None and time.time() - wait_start >= timeout:
                    return False
                time.sleep(self.check_interval_sec)
                continue
            decision = self.gate()
            if decision.action == "permit":
                self._account_wait(deferred_at, wait_start)
                return True
            if decision.action == "cancel":
                self.cancel(decision.reason)
                return False
            if deferred_at is None:
                deferred_at = time.time()
                with self._lock:
                    self._pauses += 1
                    self._last_pause_reason = decision.reason
                    self._record("defer", reason=decision.reason)
                logger.info("FL training deferred (%s)", decision.reason)
            if timeout is not None and time.time() - wait_start >= timeout:
                return False
            time.sleep(self.check_interval_sec)

    # ------------------------------------------------------------------
    def _account_wait(self, deferred_at: Optional[float], wait_start: float) -> None:
        """Fold the deferral/wait time into the counters."""
        with self._lock:
            if deferred_at is not None:
                self._paused_total_sec += time.time() - deferred_at
                self._wait_total_sec += time.time() - wait_start

    # ------------------------------------------------------------------
    def status(self) -> dict:
        """JSON-serializable snapshot for the dashboard / experiment log."""
        with self._lock:
            decision = self._last_decision
            return {
                "state": self._state,
                "started_at": self._started_at,
                "pauses": self._pauses,
                "paused_total_sec": round(self._paused_total_sec, 3),
                "wait_total_sec": round(self._wait_total_sec, 3),
                "last_pause_reason": self._last_pause_reason,
                "last_decision": decision.to_dict() if decision else None,
                "policy": asdict(self.policy.config),
                "events": list(self._events),
            }

    # ------------------------------------------------------------------
    def _record(self, event: str, reason: str = "") -> None:
        self._events.append({
            "event": event,
            "reason": reason,
            "timestamp": round(time.time(), 3),
        })


def create_controller_from_config(
    config: ResourceConfig,
    monitor: Optional[ResourceMonitor] = None,
    check_interval_sec: Optional[float] = None,
) -> TrainingController:
    """Factory: build a TrainingController from a ResourceConfig."""
    return TrainingController(
        ResourcePolicy(config),
        monitor or ResourceMonitor(),
        check_interval_sec=check_interval_sec,
    )