"""File system monitoring for new/downloaded PE files.

Safety contract
---------------
- Static analysis only. Detected files are NEVER executed and never opened
  beyond a read-only handle (2-byte MZ sniff / streaming SHA-256).
- Files are dispatched to the analysis callback only after size/mtime
  stability is confirmed (no partially-written downloads).
- Unchanged files are never re-dispatched: path + (size, mtime) metadata +
  SHA-256 combine to detect new, changed, or unchanged states.

Detection
---------
A file is a candidate if its extension is in pe_extensions (e.g. .exe/.dll/
.sys) OR its first two bytes are the PE magic 'MZ' — we do not rely on
filename extensions alone.

Background operation
--------------------
``start()`` blocks in the current thread; ``start_background()`` runs the
scan loop in a daemon thread. ``status()`` returns a JSON-serializable
snapshot that a dashboard can poll; ``write_status_file()`` persists it
atomically (the Phase-10 dashboard reads this file).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from fedshield.config import ExperimentConfig, MonitorConfig
from fedshield.logging_setup import get_logger

logger = get_logger(__name__)

PE_MAGIC = b"MZ"


@dataclass
class FileEvent:
    """A candidate PE file that passed filtering and stability checks."""

    path: Path
    sha256: str
    size: int
    extension: str
    detected_at: float  # epoch seconds
    detected_by: str = "extension"  # extension | signature | both
    mtime: float = 0.0

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size": self.size,
            "extension": self.extension,
            "detected_at": round(self.detected_at, 3),
            "detected_by": self.detected_by,
            "mtime": round(self.mtime, 3),
        }


def has_pe_signature(path: Path) -> bool:
    """True if the file starts with the PE magic 'MZ' (read-only, 2 bytes)."""
    try:
        with open(path, "rb") as f:
            return f.read(2) == PE_MAGIC
    except OSError:
        return False


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically (temp file + rename) to avoid torn reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class FileMonitor:
    """Polls watched directories for new PE candidates and dispatches them.

    ``callback`` receives a FileEvent per stable, previously-unseen file.
    ``sleep_fn``/``time_fn`` are injectable for deterministic tests.
    """

    def __init__(
        self,
        config: MonitorConfig,
        callback: Callable[[FileEvent], None],
        poll_interval: float = 1.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] = time.time,
    ):
        self.config = config
        self.callback = callback
        self.poll_interval = poll_interval
        self.sleep_fn = sleep_fn
        self.time_fn = time_fn

        self._lock = threading.Lock()
        # path -> last (mtime, size) seen; used to detect new/changed files
        self._seen: Dict[str, tuple[float, int]] = {}
        # path -> sha256 of last dispatched content; unchanged files are skipped
        self._scanned: Dict[str, str] = {}
        self._recent_events: List[dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._started_at: Optional[float] = None
        self._files_analyzed = 0
        self._files_seen = 0
        self._last_scan_at: Optional[float] = None
        self._errors = 0

    # -- status ------------------------------------------------------------

    def status(self) -> dict:
        """JSON-serializable snapshot for the dashboard."""
        with self._lock:
            return {
                "running": self._running,
                "watched_directories": [str(p) for p in self._expand_paths()],
                "recursive": self.config.recursive,
                "poll_interval": self.poll_interval,
                "started_at": self._started_at,
                "last_scan_at": self._last_scan_at,
                "files_seen": self._files_seen,
                "files_analyzed": self._files_analyzed,
                "errors": self._errors,
                "recent_events": list(reversed(self._recent_events[-10:])),
            }

    def write_status_file(self, path: Path) -> None:
        """Persist the status snapshot for out-of-process dashboards."""
        _atomic_write_json(path, self.status())

    # -- helpers -----------------------------------------------------------

    def _expand_paths(self) -> list[Path]:
        paths = []
        for d in self.config.watched_directories:
            expanded = Path(os.path.expanduser(os.path.expandvars(d)))
            if expanded.is_dir():
                paths.append(expanded)
            else:
                logger.warning("Watched directory does not exist: %s", expanded)
        return paths

    def _is_pe_extension(self, path: Path) -> bool:
        return path.suffix.lower() in tuple(e.lower() for e in self.config.pe_extensions)

    def _is_target(self, path: Path, size: int) -> bool:
        """Extension match OR real PE signature sniff (not extension-only)."""
        if self._is_pe_extension(path):
            return True
        if size < 2:
            return False
        return has_pe_signature(path)

    def _compute_sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _wait_for_stability(self, path: Path) -> Optional[tuple[float, int]]:
        """Block until size+mtime hold still for two consecutive samples.

        Returns (mtime, size) or None when the file vanished / timed out
        / never stabilized (e.g. still being written).
        """
        last = None
        deadline = self.time_fn() + self.config.stability_wait * 3
        while self.time_fn() < deadline:
            try:
                stat = path.stat()
                sample = (stat.st_mtime, stat.st_size)
            except OSError:
                return None
            if last is not None and sample == last and sample[1] > 0:
                return sample
            last = sample
            self.sleep_fn(0.05)
        return None

    # -- scanning ----------------------------------------------------------

    def _scan_tree(self, root: Path):
        """Yield files under root (recursively when enabled)."""
        if not self.config.recursive:
            for entry in root.iterdir():
                if entry.is_file():
                    yield entry
            return
        for current, dirs, files in os.walk(root):
            for name in files:
                yield Path(current) / name

    def _scan_directory(self, dir_path: Path) -> list[FileEvent]:
        events = []
        for entry in self._scan_tree(dir_path):
            if not entry.is_file():
                continue
            try:
                stat = entry.stat()
            except OSError:
                continue
            if not self._is_target(entry, stat.st_size):
                continue
            if self.config.max_file_size and stat.st_size > self.config.max_file_size:
                logger.debug("Skipping oversized file: %s (%d bytes)", entry, stat.st_size)
                continue

            key = str(entry)
            with self._lock:
                last = self._seen.get(key)
            new_or_changed = last is None or (stat.st_mtime, stat.st_size) != last

            if not new_or_changed:
                continue  # unchanged since last scan

            stable = self._wait_for_stability(entry)
            if stable is None:
                continue
            mtime, size = stable

            sha256 = self._compute_sha256(entry)
            with self._lock:
                if self._scanned.get(key) == sha256:
                    # same content as last time (e.g. touched); just refresh seen
                    self._seen[key] = (mtime, size)
                    continue

            detected_by = "extension" if self._is_pe_extension(entry) else "signature"
            if self._is_pe_extension(entry) and size >= 2 and has_pe_signature(entry):
                detected_by = "both"

            with self._lock:
                self._seen[key] = (mtime, size)
                self._scanned[key] = sha256
                self._files_seen += 1

            events.append(FileEvent(
                path=entry, sha256=sha256, size=size,
                extension=entry.suffix.lower(),
                detected_at=self.time_fn(), detected_by=detected_by, mtime=mtime,
            ))
        return events

    def run_once(self) -> int:
        """One scan cycle over all watched dirs; returns events dispatched."""
        total = 0
        for dir_path in self._expand_paths():
            for event in self._scan_directory(dir_path):
                try:
                    self.callback(event)
                except Exception as exc:  # noqa: BLE001 - a bad handler must not stop scanning
                    logger.exception("Callback failed for %s: %s", event.path, exc)
                    with self._lock:
                        self._errors += 1
                else:
                    with self._lock:
                        self._files_analyzed += 1
                        self._recent_events.append(event.to_dict())
                    logger.info("Dispatched %s (%s, %d bytes, by %s)",
                                event.path, event.sha256[:12], event.size, event.detected_by)
                total += 1
        with self._lock:
            self._last_scan_at = self.time_fn()
        return total

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Run the monitoring loop in the current thread (blocking)."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._started_at = self.time_fn()
        logger.info("File monitor started watching: %s", self.config.watched_directories)
        self._loop()

    def start_background(self) -> None:
        """Run the loop in a daemon thread (monitor keeps working)."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._started_at = self.time_fn()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="file-monitor")
        self._thread.start()
        logger.info("File monitor started in background thread")

    def _loop(self) -> None:
        try:
            while True:
                with self._lock:
                    if not self._running:
                        break
                self.run_once()
                self.sleep_fn(self.poll_interval)
        finally:
            with self._lock:
                self._running = False
            logger.info("File monitor stopped")

    def stop(self) -> None:
        """Stop the loop; joins the background thread when present."""
        with self._lock:
            self._running = False
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(self.poll_interval * 2, 1.0))
        logger.info("File monitor stopped")


def create_monitor_from_config(
    config: ExperimentConfig, callback: Callable[[FileEvent], None]
) -> FileMonitor:
    """Factory to create FileMonitor from ExperimentConfig."""
    return FileMonitor(config.endpoint.monitor, callback,
                       poll_interval=config.endpoint.monitor.poll_interval)