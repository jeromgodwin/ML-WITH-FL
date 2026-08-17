"""Tests for logging initialization and structured logging behavior."""

import json
import logging

from fedshield.logging_setup import (
    get_logger,
    level_from_string,
    log_event,
    setup_logging,
)


def test_level_from_string():
    assert level_from_string("DEBUG") == logging.DEBUG
    assert level_from_string("INFO") == logging.INFO
    assert level_from_string("warning") == logging.WARNING
    assert level_from_string("BOGUS") == logging.INFO


def test_setup_logging_creates_file(tmp_path):
    log_dir = tmp_path / "logs"
    setup_logging(level="DEBUG", log_dir=log_dir, file_output=True, structured_output=False, reset=True)
    logger = get_logger("test.logging")
    logger.info("hello from test")
    for handler in logging.getLogger().handlers:
        handler.flush()
    log_file = log_dir / "fedshield.log"
    assert log_file.exists()
    assert "hello from test" in log_file.read_text(encoding="utf-8")


def test_structured_logging_writes_jsonl(tmp_path):
    log_dir = tmp_path / "logs"
    setup_logging(level="INFO", log_dir=log_dir, file_output=False, structured_output=True, reset=True)
    logger = get_logger("test.structured")
    log_event(logger, logging.INFO, "detected", sha256="abc123", action="WARN")
    for handler in logging.getLogger().handlers:
        handler.flush()
    lines = (log_dir / "fedshield.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert lines
    record = json.loads(lines[0])
    assert record["message"] == "detected"
    assert record["fields"]["sha256"] == "abc123"
    assert record["fields"]["action"] == "WARN"


def test_logger_reuses_global_config():
    setup_logging(reset=False)
    logger = get_logger("test.reuse")
    assert logger.name == "test.reuse"
