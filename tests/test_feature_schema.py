"""Schema tests: canonical 2381-name layout matches the vendored extractor."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.federated.data.feature_schema import (  # noqa: E402
    DATA_DIRECTORIES, N_FEATURES, build_feature_names,
)
from src.federated.data.vectorize import load_ember_features_module  # noqa: E402


@pytest.fixture(scope="module")
def extractor():
    mod = load_ember_features_module()
    return mod.PEFeatureExtractor(feature_version=2, print_feature_warning=False)


def test_schema_has_2381_unique_names():
    names = build_feature_names()
    assert len(names) == N_FEATURES == 2381
    assert len(set(names)) == len(names)


def test_schema_matches_extractor_dim(extractor):
    assert len(build_feature_names()) == extractor.dim


def test_block_sizes():
    names = build_feature_names()
    groups = {
        "histogram": 256,
        "byteentropy": 256,
        "strings": 104,
        "general": 10,
        "header": 62,
        "section": 255,
        "imports": 1280,
        "exports": 128,
        "datadirectories": 30,
    }
    prefix_counts = {}
    for name in names:
        prefix = name.split("_", 1)[0]
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    for prefix, expected in groups.items():
        assert prefix_counts[prefix] == expected, f"{prefix}: {prefix_counts[prefix]} != {expected}"
    assert sum(groups.values()) == N_FEATURES


def test_datadirectory_pairs():
    names = build_feature_names()
    dd_names = [n for n in names if n.startswith("datadirectories_")]
    assert len(dd_names) == 2 * len(DATA_DIRECTORIES)
    for dd in DATA_DIRECTORIES:
        assert f"datadirectories_{dd.lower()}_size" in dd_names
        assert f"datadirectories_{dd.lower()}_vaddr" in dd_names


def test_header_numerics_present():
    names = build_feature_names()
    for n in ["header_timestamp", "header_sizeof_code", "header_sizeof_heap_commit",
              "header_major_subsystem_version"]:
        assert n in names
