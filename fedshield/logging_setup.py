"""Centralized logging configuration for all FedShield modules.

Supports console and optional file output, configurable level, and optional
structured JSON logs. Never logs raw file contents (only paths/hashes).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def level_from_string(level: str) -> int:
    """Convert a level name (e.g. 'INFO') to a logging level int."""
    return getattr(logging, level.upper(), logging.INFO)


def _structured_formatter_factory() -> logging.Formatter:
    """Formatter producing a single-line JSON record."""

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload: dict[str, Any] = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            extra = getattr(record, "fields", None)
            if isinstance(extra, dict):
                payload["fields"] = extra
            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)
            return json.dumps(payload)

    return JsonFormatter()


def setup_logging(
    level: int | str = logging.INFO,
    log_dir: str | Path = "logs",
    file_output: bool = True,
    structured_output: bool = False,
    reset: bool = False,
) -> None:
    """Configure root logging.

    Args:
        level: logging level as int or name.
        log_dir: directory for the log file (created if missing).
        file_output: also write to ``log_dir/fedshield.log``.
        structured_output: write a separate JSONL log file.
        reset: reconfigure even if already configured (mainly for tests).
    """
    global _configured
    if _configured and not reset:
        return

    level = level_from_string(level) if isinstance(level, str) else level
    log_dir = Path(log_dir)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FMT))
    root.addHandler(console)

    if file_output or structured_output:
        log_dir.mkdir(parents=True, exist_ok=True)

    if file_output:
        file_handler = logging.FileHandler(log_dir / "fedshield.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FMT))
        root.addHandler(file_handler)

    if structured_output:
        json_handler = logging.FileHandler(log_dir / "fedshield.jsonl", encoding="utf-8")
        json_handler.setFormatter(_structured_formatter_factory())
        root.addHandler(json_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a module logger; configures default logging on first use."""
    setup_logging()
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Emit a log record with structured ``fields`` (visible in JSON logs)."""
    logger.log(level, message, extra={"fields": fields})


def log_error(logger: logging.Logger, message: str, exc: Optional[Exception] = None, **fields: Any) -> None:
    """Log an error, optionally with exception traceback and structured fields."""
    if exc is not None:
        logger.exception("%s: %s", message, exc, extra={"fields": fields})
    else:
        log_event(logger, logging.ERROR, message, **fields)
