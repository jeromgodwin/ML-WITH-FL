"""Phase 6: static PE feature extraction tests.

- benign system PEs only (never executed, never opened beyond read)
- same file -> identical vector (determinism)
- strict validation: count, dtype, NaN, order, preprocessing version
- explicit ExtractionError on any failure; no silent value filling
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoint.feature_extraction import (  # noqa: E402
    ExtractionError, FeatureExtractor, FeatureVector, load_feature_schema,
    validate_schema, validate_vector,
)
from src.federated.data.feature_schema import FEATURE_NAMES, FEATURE_VERSION, N_FEATURES  # noqa: E402

BUNDLE_SCHEMA = Path("data/ember_2018_2/models/mlp-central-v1/bundle/feature_schema.json")

SYSTEM_PES = [
    Path(r"C:\Windows\System32\kernel32.dll"),
    Path(r"C:\Windows\System32\user32.dll"),
    Path(r"C:\Windows\System32\notepad.exe"),
]

pytestmark = pytest.mark.skipif(
    not any(p.exists() for p in SYSTEM_PES),
    reason="no system PE available for benign-file tests")


@pytest.fixture(scope="module")
def extractor():
    return FeatureExtractor(load_feature_schema(BUNDLE_SCHEMA))


# ---------------------------------------------------------------------------
# schema validation
# ---------------------------------------------------------------------------

def test_schema_from_bundle_is_conformant():
    schema = load_feature_schema(BUNDLE_SCHEMA)
    assert validate_schema(schema) == []
    assert schema["feature_version"] == FEATURE_VERSION
    assert len(schema["names"]) == N_FEATURES


def test_schema_wrong_count_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"feature_version": 2, "n_features": 16, "names": [' + ", ".join(f'"f{i}"' for i in range(16)) + "]}", encoding="utf-8")
    with pytest.raises(ExtractionError, match="2381"):
        FeatureExtractor(p)


def test_schema_wrong_order_rejected():
    # same names, wrong order -> must be rejected (exact order contract)
    shuffled = list(reversed(FEATURE_NAMES))
    bad = {"names": shuffled, "feature_version": FEATURE_VERSION}
    with pytest.raises(ExtractionError, match="order"):
        FeatureExtractor(bad)


def test_schema_wrong_version_rejected():
    bad = {"names": FEATURE_NAMES, "feature_version": 1}
    with pytest.raises(ExtractionError, match="version"):
        FeatureExtractor(bad)


def test_schema_unsupported_types_rejected():
    bad = {"names": FEATURE_NAMES, "feature_version": FEATURE_VERSION,
           "feature_types": ("float64",) * N_FEATURES}
    with pytest.raises(ExtractionError, match="types"):
        FeatureExtractor(bad)


# ---------------------------------------------------------------------------
# vector validation
# ---------------------------------------------------------------------------

def test_validate_vector_accepts_good():
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    assert validate_vector(vec, {"names": FEATURE_NAMES}) == []


def test_validate_vector_rejects_nan():
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    vec[3] = np.nan
    assert any("NaN" in e for e in validate_vector(vec, {"names": FEATURE_NAMES}))


def test_validate_vector_rejects_inf():
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    vec[10] = np.inf
    assert any("NaN" in e for e in validate_vector(vec, {"names": FEATURE_NAMES}))


def test_validate_vector_rejects_wrong_dtype():
    vec = np.zeros(N_FEATURES, dtype=np.float64)
    assert any("dtype" in e for e in validate_vector(vec, {"names": FEATURE_NAMES}))


def test_validate_vector_rejects_wrong_count():
    vec = np.zeros(16, dtype=np.float32)
    assert any("count" in e for e in validate_vector(vec, {"names": FEATURE_NAMES}))


# ---------------------------------------------------------------------------
# extraction on benign PEs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", SYSTEM_PES, ids=lambda p: p.name)
def test_extract_benign_pe(extractor, path):
    fv = extractor.extract(path)
    assert isinstance(fv, FeatureVector)
    assert fv.features.shape == (N_FEATURES,)
    assert fv.features.dtype == np.float32
    assert np.isfinite(fv.features).all()
    assert fv.extraction_success
    assert fv.schema_version == "ember_v2_std"
    assert fv.file_path == str(path)
    assert fv.file_size == path.stat().st_size
    assert len(fv.sha256) == 64
    assert fv.extracted_at  # ISO timestamp present


def test_extract_deterministic(extractor):
    path = SYSTEM_PES[0]
    v1 = extractor.extract(path)
    v2 = extractor.extract(path)
    assert np.array_equal(v1.features, v2.features)
    assert v1.sha256 == v2.sha256


def test_sha256_matches_hashing(extractor):
    import hashlib

    path = SYSTEM_PES[0]
    fv = extractor.extract(path)
    assert fv.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_feature_names_order_equals_canonical(extractor):
    fv = extractor.extract(SYSTEM_PES[0])
    assert fv.feature_names == FEATURE_NAMES


def test_vector_is_schema_equivalent_to_dataset_rows(extractor):
    """The extracted vector must live in the same feature space as EMBER rows:
    identical count/order/type; scale check against a dataset row."""
    import joblib

    fv = extractor.extract(SYSTEM_PES[0])
    x_mm = np.load("data/ember_2018_2/vectorized/X_train.npy", mmap_mode="r")
    row = x_mm[0]
    assert row.shape == fv.features.shape
    assert row.dtype == fv.features.dtype
    scaler = joblib.load("data/ember_2018_2/artifacts/scaler.joblib")
    scaled = fv.features / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    assert np.isfinite(scaled).all()  # preprocessed vector is inference-ready


def test_to_dict_hides_features_by_default(extractor):
    d = extractor.extract(SYSTEM_PES[0]).to_dict()
    assert "features" not in d
    assert d["n_features"] == N_FEATURES
    d2 = extractor.extract(SYSTEM_PES[0]).to_dict(expose_features=True)
    assert len(d2["features"]) == N_FEATURES


# ---------------------------------------------------------------------------
# explicit failures (no silent filling)
# ---------------------------------------------------------------------------

def test_non_pe_file_raises(extractor, tmp_path):
    f = tmp_path / "text.txt"
    f.write_text("plain text, not a PE")
    with pytest.raises(ExtractionError, match="not a PE"):
        extractor.extract(f)


def test_truncated_mz_raises(extractor, tmp_path):
    f = tmp_path / "broken.exe"
    f.write_bytes(b"MZ" + b"\x00" * 32)  # MZ but not a parseable PE
    with pytest.raises(ExtractionError, match="parse|vectorization|not a PE"):
        extractor.extract(f)


def test_missing_file_raises(extractor, tmp_path):
    with pytest.raises(ExtractionError, match="not a file"):
        extractor.extract(tmp_path / "nope.exe")


def test_nan_in_vector_raises(extractor, monkeypatch):
    import src.endpoint.feature_extraction as fe_module

    def poisoned(_bytez):
        vec = np.zeros(N_FEATURES, dtype=np.float32)
        vec[7] = np.nan
        return vec

    monkeypatch.setattr(fe_module, "_ember_extract", poisoned)
    with pytest.raises(ExtractionError, match="NaN/Inf"):
        extractor.extract(SYSTEM_PES[0])


def test_wrong_dim_from_extractor_raises(extractor, monkeypatch):
    import src.endpoint.feature_extraction as fe_module

    monkeypatch.setattr(fe_module, "_ember_extract",
                        lambda _b: np.zeros(16, dtype=np.float32))
    with pytest.raises(ExtractionError, match="count|validation"):
        extractor.extract(SYSTEM_PES[0])