"""Phase 5: endpoint file monitor tests.

Safe synthetic files only (MZ-prefixed byte blobs) — nothing is executed,
nothing touches quarantine, and real malware is never used.
"""

import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedshield.config import MonitorConfig  # noqa: E402
from src.endpoint.monitor import (  # noqa: E402
    FileEvent, FileMonitor, create_monitor_from_config, has_pe_signature,
)

PE_BLOB = b"MZ" + b"\x00" * 256 + b"static-only-test-bytes"


def _cfg(tmp_path, **kw):
    defaults = {
        "watched_directories": (str(tmp_path),),
        "recursive": False,
        "stability_wait": 0.15,
        "poll_interval": 0.05,
        "max_file_size": 10_000_000,
    }
    defaults.update(kw)
    return MonitorConfig(**defaults)


def _monitor(cfg, callback=None, poll=0.05):
    events = []
    cb = callback or events.append
    return FileMonitor(cfg, cb, poll_interval=poll), events


# ---------------------------------------------------------------------------
# detection (extension AND signature, never extension-only)
# ---------------------------------------------------------------------------

def test_mz_signature_detected_without_extension(tmp_path):
    f = tmp_path / "payload.bin"
    f.write_bytes(PE_BLOB)
    mon, events = _monitor(_cfg(tmp_path))
    assert mon.run_once() == 1
    assert len(events) == 1
    assert events[0].path == f
    assert events[0].detected_by == "signature"
    assert events[0].sha256


def test_known_extension_emitted_even_without_mz(tmp_path):
    f = tmp_path / "notreally.exe"
    f.write_bytes(b"this is not a pe")
    mon, events = _monitor(_cfg(tmp_path))
    assert mon.run_once() == 1
    assert events[0].detected_by == "extension"


def test_plain_file_ignored(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("just text")
    mon, events = _monitor(_cfg(tmp_path))
    assert mon.run_once() == 0
    assert events == []


def test_empty_file_ignored(tmp_path):
    (tmp_path / "empty.exe").write_bytes(b"")
    mon, events = _monitor(_cfg(tmp_path))
    assert mon.run_once() == 0
    assert events == []


def test_has_pe_signature_helper(tmp_path):
    f = tmp_path / "sig.bin"
    f.write_bytes(PE_BLOB)
    assert has_pe_signature(f)
    f.write_bytes(b"PE")
    assert not has_pe_signature(f)


# ---------------------------------------------------------------------------
# stability
# ---------------------------------------------------------------------------

def test_partial_write_waits_for_stability(tmp_path):
    f = tmp_path / "download.exe"
    f.write_bytes(b"MZ" + b"\x00" * 64)
    os_utime = __import__("os").utime
    os_utime(f)  # refresh mtime before first sample

    # first stability sample sees size 66; the injected sleep then grows the
    # file, so the monitor must NOT dispatch until writes have stopped
    calls = {"n": 0}
    final_size = len(PE_BLOB)

    def growing_sleep(_seconds):
        calls["n"] += 1
        if calls["n"] == 1:
            with open(f, "ab") as fh:
                fh.write(PE_BLOB[66:])
            os_utime(f)

    mon, events = _monitor(_cfg(tmp_path, stability_wait=0.3), poll=0.05)
    mon.sleep_fn = growing_sleep
    assert mon.run_once() == 1
    assert len(events) == 1
    assert events[0].size == final_size  # only the complete file was dispatched


# ---------------------------------------------------------------------------
# duplicate prevention (path + metadata + sha256)
# ---------------------------------------------------------------------------

def test_unchanged_file_not_rescanned(tmp_path):
    f = tmp_path / "app.exe"
    f.write_bytes(PE_BLOB)
    mon, events = _monitor(_cfg(tmp_path))
    assert mon.run_once() == 1
    assert mon.run_once() == 0
    assert len(events) == 1


def test_modified_content_rescanned(tmp_path):
    f = tmp_path / "app.exe"
    f.write_bytes(PE_BLOB)
    mon, events = _monitor(_cfg(tmp_path))
    assert mon.run_once() == 1
    f.write_bytes(PE_BLOB + b"CHANGED")
    os_utime = __import__("os").utime
    os_utime(f)
    assert mon.run_once() == 1
    assert len(events) == 2
    assert events[1].size == len(PE_BLOB) + 7


def test_touched_identical_content_not_rescanned(tmp_path):
    f = tmp_path / "app.exe"
    f.write_bytes(PE_BLOB)
    mon, events = _monitor(_cfg(tmp_path))
    assert mon.run_once() == 1
    f.write_bytes(PE_BLOB)  # same bytes, new mtime
    os_utime = __import__("os").utime
    os_utime(f)
    assert mon.run_once() == 0  # sha256 identical -> not re-dispatched
    assert len(events) == 1


def test_oversized_file_skipped(tmp_path):
    f = tmp_path / "big.exe"
    f.write_bytes(PE_BLOB + b"\x00" * 500)
    mon, events = _monitor(_cfg(tmp_path, max_file_size=100))
    assert mon.run_once() == 0
    assert events == []


# ---------------------------------------------------------------------------
# recursion
# ---------------------------------------------------------------------------

def test_recursive_subdirectory_detected(tmp_path):
    sub = tmp_path / "nested" / "deep"
    sub.mkdir(parents=True)
    (sub / "module.dll").write_bytes(PE_BLOB)
    mon_rec, events_rec = _monitor(_cfg(tmp_path, recursive=True))
    assert mon_rec.run_once() == 1
    mon_flat, events_flat = _monitor(_cfg(tmp_path, recursive=False))
    assert mon_flat.run_once() == 0
    assert len(events_rec) == 1
    assert events_flat == []


# ---------------------------------------------------------------------------
# lifecycle, status, robustness
# ---------------------------------------------------------------------------

def test_background_start_stop_dispatches(tmp_path):
    done = threading.Event()
    received = []

    def cb(event: FileEvent):
        received.append(event)
        done.set()

    mon, _ = _monitor(_cfg(tmp_path), callback=cb)
    mon.start_background()
    try:
        (tmp_path / "bg.exe").write_bytes(PE_BLOB)
        assert done.wait(timeout=10), "callback not triggered in time"
        assert len(received) == 1
        assert mon.status()["files_analyzed"] == 1
        assert mon.status()["running"] is True
    finally:
        mon.stop()
    assert mon.status()["running"] is False


def test_status_fields(tmp_path):
    mon, _ = _monitor(_cfg(tmp_path))
    st = mon.status()
    assert set(st) >= {"running", "watched_directories", "recursive",
                       "poll_interval", "files_seen", "files_analyzed",
                       "recent_events", "errors"}


def test_status_file_written(tmp_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("status") / "monitor" / "status.json"
    mon, _ = _monitor(_cfg(tmp_path))
    mon.write_status_file(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["running"] is False
    assert out.parent / "status.json" == out


def test_callback_error_does_not_kill_scan(tmp_path):
    def boom(_event):
        raise RuntimeError("handler bug")

    mon, _ = _monitor(_cfg(tmp_path), callback=boom)
    (tmp_path / "x.exe").write_bytes(PE_BLOB)
    assert mon.run_once() == 1  # cycle completed despite handler failure
    assert mon.status()["errors"] == 1


def test_factory_from_experiment_config(tmp_path):
    from fedshield.config import ExperimentConfig

    cfg = ExperimentConfig()
    cfg.endpoint.monitor.watched_directories = (str(tmp_path),)
    cfg.endpoint.monitor.stability_wait = 0.1
    cfg.endpoint.monitor.poll_interval = 0.05
    mon = create_monitor_from_config(cfg, lambda e: None)
    assert mon.poll_interval == 0.05
    assert mon.run_once() == 0  # empty dir scans cleanly