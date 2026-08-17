"""Endpoint extraction tests: real PE bytes -> 2381-dim vector, shape contracts."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoint.ember_features import (  # noqa: E402
    HAS_LIEF, extract_feature_vector, extract_raw_features,
)
from src.federated.data.feature_schema import N_FEATURES  # noqa: E402

SYSTEM_PES = [
    Path(r"C:\Windows\System32\kernel32.dll"),
    Path(r"C:\Windows\System32\user32.dll"),
    Path(r"C:\Windows\System32\cmd.exe"),
]

pytestmark = pytest.mark.skipif(
    not HAS_LIEF or not any(p.exists() for p in SYSTEM_PES),
    reason="lief or a system PE is unavailable")


@pytest.mark.parametrize("path", SYSTEM_PES, ids=lambda p: p.name)
def test_extract_vector_shape_and_finite(path):
    vec = extract_feature_vector(path.read_bytes())
    assert vec is not None
    assert vec.shape == (N_FEATURES,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()


def test_extract_raw_structure():
    raw = extract_raw_features(SYSTEM_PES[0].read_bytes())
    assert raw is not None
    for key in ("histogram", "byteentropy", "strings", "general", "header",
                "section", "imports", "exports", "datadirectories"):
        assert key in raw
    assert len(raw["histogram"]) == 256
    assert len(raw["byteentropy"]) == 256
    assert raw["header"]["coff"]["machine"] in ("AMD64", "I386", "ARM64")
    assert len(raw["datadirectories"]) >= 1
    assert raw["section"]["sections"], "a real PE must have sections"


def test_extract_deterministic():
    bytez = SYSTEM_PES[0].read_bytes()
    v1 = extract_feature_vector(bytez)
    v2 = extract_feature_vector(bytez)
    assert np.array_equal(v1, v2)


def test_non_pe_returns_none():
    assert extract_feature_vector(b"not a pe file at all" * 10) is None
    assert extract_raw_features(b"definitely not pe") is None


def test_empty_bytes_returns_none():
    assert extract_feature_vector(b"") is None