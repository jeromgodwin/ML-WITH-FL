"""FedShield automatic endpoint file monitor.

Runs independently of any dashboard: polls watched directories, waits for
writes to complete, detects PE files (extension OR MZ signature), de-dups
unchanged files, and sends stable candidates through static analysis
(no execution, no quarantine in this phase).

The dashboard connects by polling the status file (JSON, written
atomically every scan cycle).

Usage:
    python scripts/run_monitor.py                       # default config + bundle
    python scripts/run_monitor.py --dirs "~/Downloads,~/Desktop" --no-analyze
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedshield.config import ExperimentConfig  # noqa: E402
from fedshield.logging_setup import get_logger  # noqa: E402
from src.endpoint.detector import AutoDetector  # noqa: E402
from src.endpoint.history import HistoryStore  # noqa: E402
from src.endpoint.monitor import FileEvent, FileMonitor  # noqa: E402
from src.endpoint.notifications import NotificationService  # noqa: E402
from src.endpoint.quarantine import QuarantineManager  # noqa: E402

logger = get_logger(__name__)

DEFAULT_BUNDLE = Path("data/ember_2018_2/models/mlp-central-v1/bundle")
DEFAULT_STATUS = Path("data/monitor/status.json")
DEFAULT_DETECTIONS = Path("data/monitor/detections.jsonl")
DEFAULT_CONFIG = Path("configs/default.yaml")


def _append_detection(record: dict, path: Path) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def build_callback(cfg: ExperimentConfig, bundle_path: Path, detections_path: Path, analyze: bool):
    """Analysis callback: detector + quarantine + history + notifications."""
    detector = None
    history = None
    notifier = None
    if analyze:
        quarantine = QuarantineManager(cfg.endpoint.quarantine)
        history = HistoryStore(cfg.endpoint.history)
        notifier = NotificationService(cfg.endpoint.notifications)
        detector = AutoDetector.load(bundle_path, quarantine_manager=quarantine)
        logger.info("analysis enabled with bundle %s (model loaded once)", bundle_path)
    else:
        logger.warning("analysis disabled (--no-analyze): candidates are logged only")

    def on_event(event: FileEvent) -> None:
        if detector is None:
            record = event.to_dict()
            record["analysis"] = "skipped"
            _append_detection(record, detections_path)
            return
        result = detector.scan(event.path)
        record = result.to_dict()
        record["analysis"] = "ok" if record["verdict"] != "ERROR" else "error"
        _append_detection(record, detections_path)
        if history is not None:
            history.add(result.record)
        if notifier is not None:
            notifier.notify(result.record)
        logger.info("analyzed %s -> verdict=%s action=%s p=%.4f (scan %.1f ms)",
                    event.path, record["verdict"], record["action"],
                    record["malware_probability"], record["total_scan_ms"])

    return on_event


def main() -> None:
    parser = argparse.ArgumentParser(description="FedShield endpoint file monitor")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dirs", default=None,
                        help="comma-separated watched directories (overrides config)")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--status-file", default=str(DEFAULT_STATUS))
    parser.add_argument("--detections", default=str(DEFAULT_DETECTIONS))
    parser.add_argument("--no-analyze", action="store_true",
                        help="log candidates without model analysis")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.dirs:
        cfg.endpoint.monitor.watched_directories = tuple(
            d.strip() for d in args.dirs.split(",") if d.strip())
    cfg.endpoint.monitor.stability_wait = 1.0
    cfg.endpoint.monitor.poll_interval = 1.0

    detections_path = Path(args.detections)
    detections_path.parent.mkdir(parents=True, exist_ok=True)
    callback = build_callback(cfg, Path(args.bundle), detections_path,
                              analyze=not args.no_analyze)

    monitor = FileMonitor(cfg.endpoint.monitor, callback,
                          poll_interval=cfg.endpoint.monitor.poll_interval)
    monitor.start_background()
    status_path = Path(args.status_file)
    try:
        while monitor.status()["running"]:
            monitor.write_status_file(status_path)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
        monitor.write_status_file(status_path)
        logger.info("monitor stopped; status -> %s, detections -> %s",
                    status_path, detections_path)


if __name__ == "__main__":
    main()