"""Vectorizer tests: JSONL raw features -> float32 matrices, metadata, NaN handling."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.federated.data.vectorize import vectorize_jsonl  # noqa: E402

DIMS = {"histogram": 256, "byteentropy": 256, "datadirectories": 15}


def _strings():
    return {
        "numstrings": 4, "avlength": 6.5, "printables": 26,
        "printabledist": [1, 2, 3, 4] + [0] * 92,
        "entropy": 1.75, "paths": 0, "urls": 0, "registry": 0, "MZ": 1,
    }


def _general():
    return {
        "size": 4096, "vsize": 8192, "has_debug": 0, "exports": 1, "imports": 3,
        "has_relocations": 1, "has_resources": 0, "has_signature": 0,
        "has_tls": 0, "symbols": 0,
    }


def _header():
    return {
        "coff": {"timestamp": 1234567890, "machine": "AMD64", "characteristics": ["RELOCS_STRIPPED"]},
        "optional": {
            "subsystem": "WINDOWS_GUI", "dll_characteristics": ["HIGH_ENTROPY_VA"],
            "magic": "PE32PLUS", "major_image_version": 0, "minor_image_version": 0,
            "major_linker_version": 2, "minor_linker_version": 30,
            "major_operating_system_version": 6, "minor_operating_system_version": 0,
            "major_subsystem_version": 6, "minor_subsystem_version": 0,
            "sizeof_code": 2048, "sizeof_headers": 1024, "sizeof_heap_commit": 4096,
        },
    }


def _section():
    # NOTE: real EMBER 2018_2 records have an empty entry name; the vendored
    # official code hashes [raw_obj['entry']], which modern sklearn rejects if
    # the entry is a non-empty string. The official dataset is safe, so we keep
    # the vendored code untouched and mirror the official data here.
    return {
        "entry": "",
        "sections": [
            {"name": ".text", "size": 1024, "entropy": 5.2, "vsize": 2048,
             "props": ["MEM_READ", "MEM_EXECUTE"]},
            {"name": ".data", "size": 512, "entropy": 3.1, "vsize": 1024,
             "props": ["MEM_READ", "MEM_WRITE"]},
        ],
    }


def _imports():
    return {"kernel32.dll": ["CreateFileA", "ReadFile"], "user32.dll": ["MessageBoxA"]}


def _exports():
    return ["DllMain"]


def _datadirectories():
    return [
        {"name": "DATA_DIRECTORY.EXPORT_TABLE", "size": 128, "virtual_address": 4096},
        {"name": "DATA_DIRECTORY.IMPORT_TABLE", "size": 256, "virtual_address": 8192},
    ]


def make_record(seed=1, label=1, zero_hist=False, with_dd=True):
    rng = np.random.default_rng(seed)
    record = {
        "sha256": f"deadbeef{seed:02x}", "md5": "x", "appeared": "2018-05",
        "label": label, "avclass": "xtrat" if label else "",
        "histogram": [0] * 256 if zero_hist else rng.integers(0, 50000, 256).tolist(),
        "byteentropy": rng.integers(0, 50000, 256).tolist(),
        "strings": _strings(), "general": _general(), "header": _header(),
        "section": _section(), "imports": _imports(), "exports": _exports(),
    }
    if with_dd:
        record["datadirectories"] = _datadirectories()
    return record


@pytest.fixture()
def jsonl_file(tmp_path):
    records = [
        make_record(seed=1, label=0),
        make_record(seed=2, label=1),
        make_record(seed=3, label=1, zero_hist=True),
        make_record(seed=4, label=None),
    ]
    p = tmp_path / "raw.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def test_vectorize_shapes_and_labels(jsonl_file, tmp_path):
    out = tmp_path / "out"
    stats = vectorize_jsonl(jsonl_file, out / "train", chunk_rows=2)
    assert stats.n_rows == 4
    assert stats.n_errors == 0

    X = np.load(tmp_path / "out" / "X_train.npy")
    y = np.load(tmp_path / "out" / "y_train.npy")
    assert X.shape == (4, 2381)
    assert X.dtype == np.float32
    assert y.tolist() == [0, 1, 1, -1]

    meta = [json.loads(l) for l in
            (tmp_path / "out" / "meta_train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [m["label"] for m in meta] == [0, 1, 1, None]
    assert [m["sha256"] for m in meta] == [f"deadbeef{seed:02x}" for seed in (1, 2, 3, 4)]


def test_vectorize_nan_zero_filled(jsonl_file, tmp_path):
    stats = vectorize_jsonl(jsonl_file, tmp_path / "out" / "train", chunk_rows=1)
    X = np.load(tmp_path / "out" / "X_train.npy")
    assert np.isfinite(X).all()
    # zero_hist row: empty-file histogram normalization 0/0 must have been
    # zero-filled (histogram block = columns 0..255)
    assert stats.n_nan >= 256
    assert (X[2, :256] == 0).all()


def test_vectorize_rejects_bad_lines(jsonl_file, tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(make_record(seed=9)) + "\n" + "not json\n",
                 encoding="utf-8")
    stats = vectorize_jsonl(p, tmp_path / "out" / "train", chunk_rows=1)
    assert stats.n_rows == 2
    assert stats.n_errors == 1
    X = np.load(tmp_path / "out" / "X_train.npy")
    assert X.shape == (2, 2381)


def test_entry_string_shim_matches_character_hashing(tmp_path):
    """The shim must reproduce old-sklearn per-character entry hashing."""
    from sklearn.feature_extraction import FeatureHasher
    from src.federated.data.vectorize import _normalize_record, load_ember_features_module

    record = make_record(seed=1)
    record["section"]["entry"] = "ABCDE"  # non-empty string like the official data
    norm = _normalize_record(record)
    assert isinstance(norm["section"]["entry"], list)
    assert norm["section"]["entry"] == list("ABCDE")

    # entry_name_hashed occupies SectionInfo's offset 5+50+50+50 = 155..204
    mod = load_ember_features_module()
    shimmed = mod.SectionInfo().process_raw_features(norm["section"])

    expected = FeatureHasher(50, input_type="string").transform([list("ABCDE")]).toarray()[0]
    assert np.allclose(shimmed[155:205], expected, atol=0)


def test_section_entry_empty_string_allowed(tmp_path):
    """Empty-string entries (also present in official data) must not raise."""
    import json as _json
    from src.federated.data.vectorize import vectorize_jsonl

    record = make_record(seed=7)
    record["section"]["entry"] = ""
    p = tmp_path / "e.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(_json.dumps(record) + "\n")
    stats = vectorize_jsonl(p, tmp_path / "out" / "train", chunk_rows=1)
    assert stats.n_errors == 0
