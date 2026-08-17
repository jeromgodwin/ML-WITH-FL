"""Notifications for detection events.

Notification content is minimal and user-facing only: file, verdict, risk,
action. Internal details (feature vectors, model internals, thresholds) are
never exposed.

Channels (NotificationConfig.channels): console | file | toast.
- toast is best-effort desktop notification (winotify); if unavailable the
  channel degrades to a logged warning — it never raises.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fedshield.config import NotificationConfig
from fedshield.logging_setup import get_logger
from src.interfaces import DetectionRecord

logger = get_logger(__name__)


class NotificationService:
    def __init__(self, config: NotificationConfig | None = None):
        self.config = config or NotificationConfig()

    def should_notify(self, record: DetectionRecord) -> bool:
        if not self.config.enabled:
            return False
        min_level = self.config.min_level  # allow | warn | quarantine
        order = {"allow": 0, "warn": 1, "quarantine": 2}
        level = {"ALLOW": 0, "WARN": 1, "QUARANTINE": 2}.get(record.action, 0)
        return level >= order.get(min_level, 1)

    def notify(self, record: DetectionRecord) -> None:
        if not self.should_notify(record):
            return
        for channel in self.config.channels:
            try:
                if channel == "console":
                    self._console(record)
                elif channel == "file":
                    self._file(record)
                elif channel == "toast":
                    self._toast(record)
                elif channel == "webhook":
                    logger.warning("webhook channel not implemented; skipped for %s",
                                   record.filename)
                else:
                    logger.warning("unknown notification channel %r", channel)
            except Exception as exc:  # notifications must never crash the detector
                logger.error("notification channel %s failed: %s", channel, exc)

    # ------------------------------------------------------------------
    def _console(self, record: DetectionRecord) -> None:
        logger.warning("NOTIFICATION: %s | verdict=%s risk=%d/100 action=%s",
                       record.filename, record.verdict, record.risk_score, record.action)

    def _file(self, record: DetectionRecord) -> None:
        log_path = Path(self.config.notification_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{datetime.now().isoformat()} | {record.filename} | "
                f"verdict={record.verdict} risk={record.risk_score}/100 "
                f"action={record.action}\n")

    def _toast(self, record: DetectionRecord) -> None:
        try:
            from winotify import Notification  # type: ignore

            toast = Notification(
                app_id="FedShield",
                title=f"{record.filename}: {record.verdict}",
                msg=f"Action: {record.action} | Risk: {record.risk_score}/100",
            )
            toast.show()
        except ImportError:
            logger.warning("toast channel unavailable (install 'winotify'); "
                           "falling back to console/file only")