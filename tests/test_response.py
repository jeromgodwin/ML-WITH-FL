"""Phase 8: quarantine + race conditions + notifications + history + service.

Harmless test files only; high-risk detections are mocked. No malware is
ever used or executed.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedshield.config import (  # noqa: E402
    HistoryConfig, NotificationConfig, QuarantineConfig,
)
from src.endpoint.history import HistoryStore  # noqa: E402
from src.endpoint.notifications import NotificationService  # noqa: E402
from src.endpoint.quarantine import QuarantineError, QuarantineManager  # noqa: E402
from src.endpoint.service import DashboardService  # noqa: E402
from src.interfaces import DetectionRecord  # noqa: E402


def make_record(action: str = "QUARANTINE", p: float = 0.9, name: str = "x.exe",
                risk_level: str = "HIGH", verdict: str = "HIGH",
                model_version: str = "mlp-central-v1",
                detection_id: str = "det123", timestamp: str = "2026-08-15T12:00:00+00:00") -> DetectionRecord:
    return DetectionRecord(
        detection_id=detection_id,
        timestamp=timestamp,
        filename=name,
        filepath=f"C:/tmp/{name}",
        sha256="ab" * 32,
        file_type="pe_exe",
        model_version=model_version,
        malware_probability=p,
        benign_probability=round(1 - p, 6),
        risk_score=int(p * 100),
        risk_level=risk_level,
        verdict=verdict,
        action=action,
        model_algorithm="centralized",
        analysis_duration_ms=3.5,
    )


# ---------------------------------------------------------------------------
# quarantine
# ---------------------------------------------------------------------------

def test_quarantine_moves_and_records(tmp_path):
    src = tmp_path / "sample.exe"
    src.write_bytes(b"MZ" + b"\x00" * 64)
    qdir = tmp_path / "quarantine"
    mgr = QuarantineManager(QuarantineConfig(quarantine_dir=str(qdir), preserve_filename=True))

    qr = mgr.quarantine(src, make_record(name="sample.exe"))

    assert not src.exists()
    dest = Path(qr.quarantine_path)
    assert dest.exists()
    assert dest.name == "det123_sample.exe"  # original filename preserved
    assert qr.original_path == str(src)       # original path preserved
    assert qr.sha256 == "ab" * 32
    assert qr.detection.model_version == "mlp-central-v1"
    assert qr.detection.verdict == "HIGH"
    assert qr.detection.risk_score == 90
    assert qr.detection.action == "QUARANTINE"
    assert qr.detection.timestamp
    # persistent index
    assert (qdir / "quarantine_records.jsonl").exists()
    assert len(mgr.list()) == 1


def test_quarantine_never_deletes_original(tmp_path):
    src = tmp_path / "keep.exe"
    src.write_bytes(b"MZ" + b"\x00" * 64)
    mgr = QuarantineManager(QuarantineConfig(quarantine_dir=str(tmp_path / "q")))
    mgr.quarantine(src, make_record(name="keep.exe"))
    # nothing was deleted: the file still exists inside the quarantine dir
    assert list((tmp_path / "q").glob("*.exe"))


def test_restore_moves_back(tmp_path):
    src = tmp_path / "r.exe"
    src.write_bytes(b"MZ" + b"\x00" * 64)
    mgr = QuarantineManager(QuarantineConfig(quarantine_dir=str(tmp_path / "q")))
    qr = mgr.quarantine(src, make_record(name="r.exe"))
    restored = mgr.restore(Path(qr.quarantine_path))
    assert restored == src
    assert src.exists()
    assert not Path(qr.quarantine_path).exists()


# --- race conditions -------------------------------------------------------

def test_quarantine_missing_file_controlled_error(tmp_path):
    mgr = QuarantineManager(QuarantineConfig(quarantine_dir=str(tmp_path / "q")))
    with pytest.raises(QuarantineError) as ei:
        mgr.quarantine(tmp_path / "ghost.exe", make_record(name="ghost.exe"))
    assert ei.value.reason == "missing"


def test_quarantine_disappearing_file_controlled_error(tmp_path, monkeypatch):
    src = tmp_path / "gone.exe"
    src.write_bytes(b"MZ" + b"\x00" * 64)
    mgr = QuarantineManager(QuarantineConfig(quarantine_dir=str(tmp_path / "q")))

    import shutil

    real_move = shutil.move

    def deleting_move(s, d):
        real_move(s, d)
        Path(s).unlink()  # simulate: file vanishes between exists() and move()

    monkeypatch.setattr(shutil, "move", deleting_move)
    with pytest.raises(QuarantineError) as ei:
        mgr.quarantine(src, make_record(name="gone.exe"))
    assert ei.value.reason == "missing"


def test_quarantine_locked_file_controlled_error(tmp_path, monkeypatch):
    src = tmp_path / "locked.exe"
    src.write_bytes(b"MZ" + b"\x00" * 64)
    mgr = QuarantineManager(QuarantineConfig(quarantine_dir=str(tmp_path / "q")))

    import shutil

    def locked_move(s, d):
        raise PermissionError(13, "file locked by another process", s)

    monkeypatch.setattr(shutil, "move", locked_move)
    with pytest.raises(QuarantineError) as ei:
        mgr.quarantine(src, make_record(name="locked.exe"))
    assert ei.value.reason == "locked"
    assert src.exists()  # untouched


def test_quarantine_destination_unavailable(tmp_path):
    src = tmp_path / "d.exe"
    src.write_bytes(b"MZ" + b"\x00" * 64)
    blocker = tmp_path / "q"
    blocker.write_text("I am a file, not a directory")  # mkdir will fail
    mgr = QuarantineManager(QuarantineConfig(quarantine_dir=str(blocker)))
    with pytest.raises(QuarantineError) as ei:
        mgr.quarantine(src, make_record(name="d.exe"))
    assert ei.value.reason == "destination_unavailable"


def test_duplicate_quarantine_destination_controlled_error(tmp_path):
    # same filename arriving again (duplicate event) -> same dest -> controlled error
    d1 = tmp_path / "in1"
    d2 = tmp_path / "in2"
    d1.mkdir()
    d2.mkdir()
    src = d1 / "dup.exe"
    src.write_bytes(b"MZ" + b"\x00" * 64)
    qdir = tmp_path / "q"
    mgr = QuarantineManager(QuarantineConfig(quarantine_dir=str(qdir)))
    mgr.quarantine(src, make_record(name="dup.exe"))
    src2 = d2 / "dup.exe"
    src2.write_bytes(b"MZ" + b"\x00" * 64)
    with pytest.raises(QuarantineError) as ei:
        mgr.quarantine(src2, make_record(name="dup.exe"))
    assert ei.value.reason == "duplicate"
    assert src2.exists()  # second file untouched


def test_locked_file_retries_then_succeeds(tmp_path, monkeypatch):
    """Transient lock (first attempt fails) must succeed after retry."""
    src = tmp_path / "slow.exe"
    src.write_bytes(b"MZ" + b"\x00" * 64)
    qdir = tmp_path / "q"
    mgr = QuarantineManager(QuarantineConfig(quarantine_dir=str(qdir)))

    import shutil

    real_move = shutil.move
    state = {"calls": 0}

    def flaky_move(s, d):
        state["calls"] += 1
        if state["calls"] == 1:
            raise PermissionError(13, "still being written", s)
        return real_move(s, d)

    monkeypatch.setattr(shutil, "move", flaky_move)
    qr = mgr.quarantine(src, make_record(name="slow.exe"))
    assert Path(qr.quarantine_path).exists()
    assert state["calls"] == 2


def test_detector_does_not_crash_on_quarantine_failure(tmp_path, monkeypatch):
    """Monitor must keep running even when quarantine is impossible."""
    import shutil

    from src.endpoint.detector import AutoDetector
    from src.endpoint.feature_extraction import FeatureVector
    from src.federated.data.feature_schema import FEATURE_NAMES, N_FEATURES
    import numpy as np

    class FakeBundle:
        manifest = {"version": "v1", "algorithm": "fake"}

        def predict_proba(self, X):
            return np.full(X.shape[0], 0.95, dtype=np.float32)

    class FakeExtractor:
        def extract(self, path):
            return FeatureVector(
                features=np.zeros(N_FEATURES, dtype=np.float32),
                feature_names=FEATURE_NAMES, schema_version="v2",
                model_version="v1", extraction_success=True,
                missing_features=[], extra_features=[],
                sha256="0" * 64, file_path=str(path), file_size=1,
                extracted_at="2026-01-01T00:00:00+00:00")

    src = tmp_path / "evil.exe"
    src.write_bytes(b"MZ" + b"\x00" * 64)
    qdir = tmp_path / "q"
    qdir.write_text("blocked")  # quarantine destination unusable

    mgr = QuarantineManager(QuarantineConfig(quarantine_dir=str(qdir)))
    d = AutoDetector(FakeBundle(), FakeExtractor(), quarantine_manager=mgr)
    result = d.scan(src)
    assert result.record.verdict == "HIGH"
    assert result.record.action == "QUARANTINE"  # detection recorded regardless
    assert src.exists()  # file untouched, no crash, controlled failure logged


# ---------------------------------------------------------------------------
# notifications
# ---------------------------------------------------------------------------

class FakeRecord:
    pass


def test_notification_minimal_fields(tmp_path):
    log = tmp_path / "notifications.log"
    svc = NotificationService(NotificationConfig(
        enabled=True, channels=("file",), notification_log=str(log), min_level="warn"))
    svc.notify(make_record(name="evil.exe"))
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 1
    line = lines[0]
    assert "evil.exe" in line and "HIGH" in line and "90/100" in line
    assert "QUARANTINE" in line
    assert "features" not in line.lower()  # no internal details


def test_notification_respects_min_level(tmp_path):
    log = tmp_path / "n.log"
    svc = NotificationService(NotificationConfig(
        enabled=True, channels=("file",), notification_log=str(log), min_level="quarantine"))
    svc.notify(make_record(action="ALLOW", p=0.1, verdict="LOW", risk_level="LOW",
                           name="ok.exe"))
    assert not log.exists() or log.read_text() == ""  # ALLOW not notified at quarantine level


def test_notification_never_raises(tmp_path, monkeypatch):
    svc = NotificationService(NotificationConfig(
        enabled=True, channels=("toast", "file"),
        notification_log=str(tmp_path / "n.log"), min_level="allow"))

    def boom(*a, **k):
        raise RuntimeError("toast subsystem broken")

    monkeypatch.setattr(svc, "_toast", boom)
    svc.notify(make_record(name="z.exe"))  # must not raise


# ---------------------------------------------------------------------------
# history (SQLite)
# ---------------------------------------------------------------------------

def make_history(tmp_path, retention_days=0) -> HistoryStore:
    return HistoryStore(HistoryConfig(history_db=str(tmp_path / "h.db"),
                                      retention_days=retention_days))


def test_history_add_and_latest(tmp_path):
    hs = make_history(tmp_path)
    hs.add(make_record(name="a.exe", p=0.9, detection_id="det1",
                       timestamp="2026-08-15T12:00:00+00:00"))
    hs.add(make_record(name="b.exe", p=0.1, action="ALLOW", verdict="LOW",
                       risk_level="LOW", detection_id="det2",
                       timestamp="2026-08-15T13:00:00+00:00"))
    rows = hs.latest()
    assert len(rows) == 2
    assert rows[0]["timestamp"] > rows[1]["timestamp"]  # DESC order
    assert hs.count() == 2


def test_history_blocked_and_safe(tmp_path):
    hs = make_history(tmp_path)
    hs.add(make_record(name="q.exe", detection_id="det1"))
    hs.add(make_record(name="s.exe", p=0.01, action="ALLOW", verdict="LOW",
                       risk_level="LOW", detection_id="det2"))
    blocked = hs.quarantined()
    safe = hs.safe()
    assert [r["filename"] for r in blocked] == ["q.exe"]
    assert [r["filename"] for r in safe] == ["s.exe"]


def test_history_search_filters(tmp_path):
    hs = make_history(tmp_path)
    hs.add(make_record(name="old.exe", model_version="mlp-v0", detection_id="det1",
                       timestamp="2026-01-01T00:00:00+00:00"))
    hs.add(make_record(name="new.exe", model_version="mlp-central-v1", detection_id="det2",
                       timestamp="2026-08-15T00:00:00+00:00"))
    assert [r["filename"] for r in hs.search(model_version="mlp-central-v1")] == ["new.exe"]
    assert [r["filename"] for r in hs.search(since="2026-06-01T00:00:00+00:00")] == ["new.exe"]
    assert [r["filename"] for r in hs.search(until="2026-02-01T00:00:00+00:00")] == ["old.exe"]
    assert [r["filename"] for r in hs.search(min_risk_score=80)] == ["new.exe", "old.exe"]
    assert [r["filename"] for r in hs.search(verdict="LOW")] == []


def test_history_retention_prunes(tmp_path):
    db = str(tmp_path / "h.db")
    hs = HistoryStore(HistoryConfig(history_db=db, retention_days=30))
    hs.add(make_record(name="old.exe", detection_id="det1",
                       timestamp="2026-01-01T00:00:00+00:00"))
    hs.add(make_record(name="new.exe", detection_id="det2",
                       timestamp="2026-08-15T00:00:00+00:00"))
    # re-open: prune runs on init and removes rows older than retention window
    hs2 = HistoryStore(HistoryConfig(history_db=db, retention_days=30))
    assert [r["filename"] for r in hs2.latest()] == ["new.exe"]


def test_history_roundtrip_all_fields(tmp_path):
    hs = make_history(tmp_path)
    rec = make_record(name="full.exe")
    hs.add(rec)
    row = hs.latest()[0]
    assert row["detection_id"] == rec.detection_id
    assert row["sha256"] == rec.sha256
    assert row["analysis_duration_ms"] == pytest.approx(3.5)
    assert row["benign_probability"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# dashboard service interface
# ---------------------------------------------------------------------------

def test_dashboard_service_contract(tmp_path):
    hs = make_history(tmp_path)
    hs.add(make_record(name="a.exe"))
    qdir = tmp_path / "q"
    qm = QuarantineManager(QuarantineConfig(quarantine_dir=str(qdir)))

    src = tmp_path / "z.exe"
    src.write_bytes(b"MZ" + b"\x00" * 64)
    qm.quarantine(src, make_record(name="z.exe"))

    svc = DashboardService(history=hs, quarantine=qm)
    status = svc.get_status()
    assert status["history_entries"] == 1
    assert status["quarantined_count"] == 1
    dets = svc.get_detections()
    assert len(dets) == 1 and dets[0]["filename"] == "a.exe"
    qs = svc.get_quarantined()
    assert len(qs) == 1
    assert qs[0]["detection"]["filename"] == "z.exe"
    # summary must not leak internals
    d = qs[0]["detection"]
    assert "features" not in d
    assert d["model_version"] == "mlp-central-v1"
    assert svc.search_history(verdict="LOW") == []
    assert svc.get_models() == []  # no registry attached
    assert svc.get_status()["monitor"]["running"] is False  # no monitor attached


def test_dashboard_service_serializable(tmp_path):
    hs = make_history(tmp_path)
    hs.add(make_record(name="s.exe"))
    svc = DashboardService(history=hs)
    import json

    payload = json.dumps(svc.get_detections())  # must be JSON-safe
    assert "s.exe" in payload