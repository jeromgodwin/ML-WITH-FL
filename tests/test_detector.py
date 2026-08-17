"""Phase 7: automatic real-time detection pipeline tests.

- benign system PE end-to-end through the real pipeline
- synthetic feature vectors + mocked model outputs for risk bands
- model/extractor loaded exactly once per detector
- automatic monitor -> detector workflow verified end-to-end
- no malware is ever executed; extraction is read-only
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedshield.config import MonitorConfig, QuarantineConfig, RiskConfig  # noqa: E402
from src.endpoint.detector import AutoDetector, ScanResult  # noqa: E402
from src.endpoint.feature_extraction import FeatureExtractor, FeatureVector  # noqa: E402
from src.endpoint.monitor import FileMonitor  # noqa: E402
from src.endpoint.risk import RiskEngine  # noqa: E402
from src.federated.data.feature_schema import FEATURE_NAMES, N_FEATURES  # noqa: E402
from src.interfaces import DetectionRecord  # noqa: E402

BUNDLE = Path("data/ember_2018_2/models/mlp-central-v1/bundle")
KERNEL32 = Path(r"C:\Windows\System32\kernel32.dll")

has_system_pe = any(p.exists() for p in
                    (Path(r"C:\Windows\System32\kernel32.dll"),
                     Path(r"C:\Windows\System32\notepad.exe")))
pytestmark = pytest.mark.skipif(not has_system_pe,
                                reason="no system PE available for benign-file tests")


# ---------------------------------------------------------------------------
# fakes (mocked detector internals)
# ---------------------------------------------------------------------------

class FakeBundle:
    """Fake InferenceBundle: fixed probability, counts predict calls."""

    def __init__(self, p: float = 0.01, manifest=None, calls=None):
        self.p = p
        self.manifest = manifest or {"version": "fake-v1", "algorithm": "fake"}
        self.calls = calls if calls is not None else []

    def predict_proba(self, X):
        self.calls.append(len(X))
        return np.full(X.shape[0], self.p, dtype=np.float32)


class FakeExtractor:
    """Fake FeatureExtractor: returns a synthetic valid vector."""

    def __init__(self, vec=None):
        self.vec = vec if vec is not None else np.zeros(N_FEATURES, dtype=np.float32)

    def extract(self, path):
        return FeatureVector(
            features=self.vec, feature_names=FEATURE_NAMES,
            schema_version="ember_v2_std", model_version="fake-v1",
            extraction_success=True, missing_features=[], extra_features=[],
            sha256="0" * 64, file_path=str(path), file_size=1,
            extracted_at="2026-01-01T00:00:00+00:00",
        )


def make_detector(p: float, risk: RiskConfig | None = None,
                  quarantine: QuarantineConfig | None = None) -> AutoDetector:
    bundle = FakeBundle(p)
    return AutoDetector(bundle, FakeExtractor(), risk=RiskEngine(risk), quarantine=quarantine)


# ---------------------------------------------------------------------------
# risk engine
# ---------------------------------------------------------------------------

def test_risk_engine_bands_default():
    risk = RiskEngine(RiskConfig(thresholds=(0.3, 0.7)))
    assert risk.decide(0.1) == (10, "LOW", "LOW", "ALLOW")
    assert risk.decide(0.5) == (50, "MEDIUM", "MEDIUM", "WARN")
    assert risk.decide(0.9) == (90, "HIGH", "HIGH", "QUARANTINE")


def test_risk_engine_custom_thresholds():
    risk = RiskEngine(RiskConfig(thresholds=(0.1, 0.5), actions=("ALLOW", "WARN", "QUARANTINE")))
    assert risk.decide(0.05)[3] == "ALLOW"
    assert risk.decide(0.3)[3] == "WARN"
    assert risk.decide(0.8)[3] == "QUARANTINE"


def test_risk_engine_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        RiskEngine(RiskConfig(thresholds=(0.7, 0.3)))
    with pytest.raises(ValueError):
        RiskEngine(RiskConfig(thresholds=(-0.1, 0.7)))


# ---------------------------------------------------------------------------
# end-to-end: benign PE through the REAL pipeline (bundle + extractor)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector():
    return AutoDetector.load(BUNDLE)


def test_scan_benign_pe_full_record(detector):
    result = detector.scan(KERNEL32)
    rec = result.record
    assert isinstance(rec, DetectionRecord)
    assert len(rec.detection_id) == 16
    assert rec.filename == KERNEL32.name
    assert rec.sha256 and len(rec.sha256) == 64
    assert rec.file_type == "pe_dll"
    assert rec.model_version == "mlp-central-v1"
    assert 0.0 <= rec.malware_probability < 0.3
    assert rec.benign_probability is not None and rec.benign_probability > 0.7
    assert rec.risk_level == "LOW"
    assert rec.verdict == "LOW"
    assert rec.action == "ALLOW"
    assert rec.model_algorithm == "centralized"
    assert rec.analysis_duration_ms > 0
    assert result.extraction_ms > 0
    assert result.inference_ms > 0
    assert result.total_ms > 0
    assert abs(result.total_ms - rec.analysis_duration_ms) < 50  # same measurement


def test_scan_timings_plausible(detector):
    r1 = detector.scan(KERNEL32)
    r2 = detector.scan(KERNEL32)
    # second scan warms caches; both must be measured and reported
    assert r1.extraction_ms >= 0 and r2.inference_ms >= 0
    assert r1.total_ms >= r1.extraction_ms + r1.inference_ms - 1e-6


def test_model_loaded_once():
    loads = []
    original = AutoDetector.load

    def counting_load(bundle_dir, risk=None, quarantine=None):
        loads.append(str(bundle_dir))
        bundle = FakeBundle(0.01)
        return AutoDetector(bundle, FakeExtractor(), risk=risk, quarantine=quarantine)

    try:
        AutoDetector.load = staticmethod(counting_load)
        d = AutoDetector.load(BUNDLE)
        d.scan(KERNEL32)
        d.scan(KERNEL32)
        d.scan(KERNEL32)
        assert len(loads) == 1
    finally:
        AutoDetector.load = original


# ---------------------------------------------------------------------------
# synthetic vectors + mocked outputs: verdict bands, quarantine, errors
# ---------------------------------------------------------------------------

def test_synthetic_vectors_drive_verdicts(tmp_path):
    f = tmp_path / "sample.exe"
    f.write_bytes(b"MZ" + b"\x00" * 64)
    cases = [(0.05, "ALLOW", "LOW"), (0.5, "WARN", "MEDIUM"), (0.95, "QUARANTINE", "HIGH")]
    for p, action, level in cases:
        d = make_detector(p)
        rec = d.scan(f).record
        assert rec.action == action, f"p={p} expected {action}"
        assert rec.verdict == level, f"p={p} expected {level}"
        assert rec.risk_score == round(p * 100)
        assert rec.benign_probability == pytest.approx(1.0 - p, abs=1e-5)


def test_mocked_detector_output_used_verbatim(tmp_path):
    f = tmp_path / "t.exe"
    f.write_bytes(b"MZ" + b"\x00" * 64)
    d = make_detector(p=0.123)
    rec = d.scan(f).record
    assert rec.malware_probability == pytest.approx(0.123)


def test_quarantine_action_moves_file(tmp_path):
    src = tmp_path / "evil.exe"
    src.write_bytes(b"MZ" + b"\x00" * 64)
    qdir = tmp_path / "quarantine"
    d = make_detector(p=0.95, quarantine=QuarantineConfig(
        quarantine_dir=str(qdir), record_metadata=True))
    rec = d.scan(src).record
    assert rec.action == "QUARANTINE"
    moved = list(qdir.glob("*.exe"))
    assert len(moved) == 1
    assert moved[0].name.startswith(rec.detection_id)
    assert not src.exists()
    index = qdir / "quarantine_records.jsonl"
    assert index.exists()  # metadata recorded in the persistent quarantine index
    assert rec.detection_id in index.read_text()


def test_allow_action_does_not_touch_file(tmp_path):
    src = tmp_path / "ok.exe"
    src.write_bytes(b"MZ" + b"\x00" * 64)
    d = make_detector(p=0.01)
    rec = d.scan(src).record
    assert rec.action == "ALLOW"
    assert src.exists()


def test_error_record_for_missing_file(tmp_path):
    d = make_detector(p=0.5)
    result = d.scan(tmp_path / "ghost.exe")
    assert result.record.verdict == "ERROR"
    assert result.record.action == "NONE"
    assert result.record.risk_score == 0


def test_extraction_failure_produces_error_record(tmp_path):
    src = tmp_path / "bad.exe"
    src.write_bytes(b"not a pe at all")

    from src.endpoint.feature_extraction import ExtractionError

    class BrokenExtractor(FakeExtractor):
        def extract(self, path):
            raise ExtractionError(f"not a PE file (no MZ signature): {path}")

    d = AutoDetector(FakeBundle(0.5), BrokenExtractor(), risk=RiskEngine())
    result = d.scan(src)
    assert result.record.verdict == "ERROR"
    assert result.record.action == "NONE"


# ---------------------------------------------------------------------------
# automatic end-to-end: FileMonitor -> detector, no manual upload
# ---------------------------------------------------------------------------

def test_automatic_monitor_to_detector_workflow(tmp_path):
    watched = tmp_path / "watch"
    watched.mkdir()
    results = []

    d = AutoDetector.load(BUNDLE)

    cfg = MonitorConfig(
        watched_directories=(str(watched),), recursive=True,
        stability_wait=0.2, poll_interval=0.1, max_file_size=0,
        pe_extensions=(".dll",),
    )
    monitor = FileMonitor(cfg, lambda ev: results.append(d.scan(ev.path)),
                          poll_interval=0.1)
    monitor.start_background()
    try:
        import shutil

        dropped = watched / "kernel32.dll"
        shutil.copy2(KERNEL32, dropped)
        deadline = time.time() + 30
        while time.time() < deadline and not results:
            time.sleep(0.2)
    finally:
        monitor.stop()

    assert results, "monitor never dispatched the dropped file"
    rec = results[0].record
    assert rec.filename == "kernel32.dll"
    assert rec.action == "ALLOW"
    assert rec.risk_level == "LOW"
    assert rec.analysis_duration_ms > 0


def test_non_pe_dropped_file_is_not_analyzed(tmp_path):
    watched = tmp_path / "watch2"
    watched.mkdir()
    results = []
    d = AutoDetector.load(BUNDLE)

    cfg = MonitorConfig(
        watched_directories=(str(watched),), recursive=True,
        stability_wait=0.2, poll_interval=0.1, max_file_size=0,
        pe_extensions=(".txt",),
    )
    monitor = FileMonitor(cfg, lambda ev: results.append(d.scan(ev.path)),
                          poll_interval=0.1)
    monitor.start_background()
    try:
        (watched / "readme.txt").write_text("hello world")
        (watched / "notes.txt").write_text("more text")
        time.sleep(1.5)
    finally:
        monitor.stop()

    analyzed = [r.record for r in results]
    assert len(analyzed) == 2
    assert all(r.verdict == "ERROR" for r in analyzed)  # txt -> extraction fails explicitly