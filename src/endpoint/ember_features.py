"""EMBER feature extraction from real PE bytes (lief 1.0 adapter).

The official EMBER feature extractor (scripts/ember_ref/features.py) was
written for lief 0.9/0.10-era APIs. lief 1.0 removed the old exception names
(lief.bad_format, ...) and changed some enum stringifications (e.g.
dll_characteristics print raw values instead of names, exported_functions are
objects instead of strings). This module reimplements the lief-dependent raw
feature blocks against lief 1.0 while producing the SAME raw dict structure as
the official extractor, so the official process_raw_features (vectorization,
hashing) can be reused verbatim for identical 2381-dim vectors.

Known limitation (documented, not hidden): the official 2018_2 features were
computed with lief 0.9.0; hashed enum-name features may differ slightly between
lief versions for the same binary. Shapes and formulas are identical.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

import numpy as np

from fedshield.logging_setup import get_logger
from src.federated.data.vectorize import (
    _normalize_record,
    load_ember_features_module,
)

logger = get_logger(__name__)

try:
    import lief
    HAS_LIEF = True
except ImportError:  # pragma: no cover
    HAS_LIEF = False

# lief 1.0 exception classes (lief.lief_errors.*); old names used by the
# vendored code map onto these.
_LIEF_ERRORS = tuple(
    getattr(lief.lief_errors, name)
    for name in (
        "file_format_error", "file_error", "parsing_error", "read_error",
        "read_out_of_bound", "not_found", "corrupted", "not_implemented",
    )
    if HAS_LIEF and hasattr(lief, "lief_errors") and hasattr(lief.lief_errors, name)
) or (Exception,)


# --- Pure-python feature blocks, replicated verbatim from the official       ---
# --- extractor (lief-independent). Kept local so the vendored reference      ---
# --- file stays untouched (numpy 2.x removed np.int used by the vendored     ---
# --- ByteEntropyHistogram.raw_features).                                      ---

def _raw_histogram(bytez: bytes):
    counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
    return counts.tolist()


def _raw_byteentropy(bytez: bytes, step: int = 1024, window: int = 2048):
    a = np.frombuffer(bytez, dtype=np.uint8)
    output = np.zeros((16, 16), dtype=np.int64)
    if a.shape[0] < window:
        blocks = [a]
    else:
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step, :]
    for block in blocks:
        c = np.bincount(block >> 4, minlength=16)
        p = c.astype(np.float32) / window
        wh = np.where(c)[0]
        H = np.sum(-p[wh] * np.log2(p[wh])) * 2
        hbin = int(H * 2)
        if hbin == 16:
            hbin = 15
        output[hbin, :] += c
    return output.flatten().tolist()


_STRINGS_PATTERNS = {
    "allstrings": re.compile(b"[\x20-\x7f]{5,}"),
    "paths": re.compile(b"c:\\\\", re.IGNORECASE),
    "urls": re.compile(b"https?://", re.IGNORECASE),
    "registry": re.compile(b"HKEY_"),
    "mz": re.compile(b"MZ"),
}


def _raw_strings(bytez: bytes):
    allstrings = _STRINGS_PATTERNS["allstrings"].findall(bytez)
    if allstrings:
        string_lengths = [len(s) for s in allstrings]
        avlength = sum(string_lengths) / len(string_lengths)
        as_shifted_string = [b - ord(b"\x20") for b in b"".join(allstrings)]
        c = np.bincount(as_shifted_string, minlength=96)
        csum = c.sum()
        p = c.astype(np.float32) / csum
        wh = np.where(c)[0]
        H = float(np.sum(-p[wh] * np.log2(p[wh])))
    else:
        avlength = 0
        c = np.zeros((96,), dtype=np.float32)
        H = 0.0
        csum = 0
    return {
        "numstrings": len(allstrings),
        "avlength": avlength,
        "printabledist": c.tolist(),
        "printables": int(csum),
        "entropy": H,
        "paths": len(_STRINGS_PATTERNS["paths"].findall(bytez)),
        "urls": len(_STRINGS_PATTERNS["urls"].findall(bytez)),
        "registry": len(_STRINGS_PATTERNS["registry"].findall(bytez)),
        "MZ": len(_STRINGS_PATTERNS["mz"].findall(bytez)),
    }


def _str_enum(value: Any) -> str:
    """Enum string in the official raw-feature format, e.g. 'AMD64'."""
    return str(value).split(".")[-1]


def _raw_general(bytez: bytes, binary) -> Dict[str, Any]:
    return {
        "size": len(bytez),
        "vsize": int(binary.virtual_size),
        "has_debug": int(bool(binary.has_debug)),
        "exports": len(binary.exported_functions),
        "imports": len(binary.imported_functions),
        "has_relocations": int(bool(binary.has_relocations)),
        "has_resources": int(bool(binary.has_resources)),
        "has_signature": int(bool(binary.has_signatures)),
        "has_tls": int(bool(binary.has_tls)),
        "symbols": len(binary.symbols),
    }


def _raw_header(binary) -> Dict[str, Any]:
    header = binary.header
    optional = binary.optional_header
    return {
        "coff": {
            "timestamp": int(header.time_date_stamps),
            "machine": _str_enum(header.machine),
            "characteristics": [_str_enum(c) for c in header.characteristics_list],
        },
        "optional": {
            "subsystem": _str_enum(optional.subsystem),
            # lief 1.0 prints raw values for these; .name gives the enum name
            "dll_characteristics": [str(c.name) for c in optional.dll_characteristics_lists],
            "magic": _str_enum(optional.magic),
            "major_image_version": int(optional.major_image_version),
            "minor_image_version": int(optional.minor_image_version),
            "major_linker_version": int(optional.major_linker_version),
            "minor_linker_version": int(optional.minor_linker_version),
            "major_operating_system_version": int(optional.major_operating_system_version),
            "minor_operating_system_version": int(optional.minor_operating_system_version),
            "major_subsystem_version": int(optional.major_subsystem_version),
            "minor_subsystem_version": int(optional.minor_subsystem_version),
            "sizeof_code": int(optional.sizeof_code),
            "sizeof_headers": int(optional.sizeof_headers),
            "sizeof_heap_commit": int(optional.sizeof_heap_commit),
        },
    }


def _raw_section(binary) -> Dict[str, Any]:
    entry = ""
    try:
        section = binary.section_from_rva(binary.entrypoint - binary.imagebase)
        if section is not None:
            entry = section.name
    except Exception:  # noqa: BLE001 - mirror official fallback
        entry = ""
    if not entry:
        for s in binary.sections:
            if any(str(c).split(".")[-1] == "MEM_EXECUTE" for c in s.characteristics_lists):
                entry = s.name
                break
    sections = [
        {
            "name": s.name,
            "size": int(s.size),
            "entropy": float(s.entropy),
            "vsize": int(s.virtual_size),
            "props": [_str_enum(c) for c in s.characteristics_lists],
        }
        for s in binary.sections
    ]
    return {"entry": entry, "sections": sections}


def _raw_imports(binary) -> Dict[str, Any]:
    imports: Dict[str, List[str]] = {}
    for lib in binary.imports:
        if lib.name not in imports:
            imports[lib.name] = []
        for entry in lib.entries:
            if entry.is_ordinal:
                imports[lib.name].append("ordinal" + str(entry.ordinal))
            else:
                imports[lib.name].append(entry.name[:10000])
    return imports


def _raw_exports(binary) -> List[str]:
    return [str(e.name)[:10000] for e in binary.exported_functions]


def _raw_datadirectories(binary) -> List[Dict[str, Any]]:
    output = []
    for dd in binary.data_directories:
        output.append({
            "name": str(dd.type).replace("DATA_DIRECTORY.", ""),
            "size": int(dd.size),
            "virtual_address": int(dd.rva),
        })
    return output


def extract_raw_features(bytez: bytes) -> Optional[Dict[str, Any]]:
    """Extract the official EMBER raw feature dict from PE bytes (lief 1.0).

    Returns None if the bytes cannot be parsed as a PE (the caller decides
    whether that is an error). Parsing errors are logged at debug level.
    """
    if not HAS_LIEF:
        raise RuntimeError("lief is required for real-PE feature extraction")
    try:
        binary = lief.PE.parse(list(bytez))  # lief 1.0 expects a list of ints
    except _LIEF_ERRORS as exc:
        logger.debug("lief parse error: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 - catch-all mirrors official code
        logger.debug("lief unexpected error: %s", exc)
        return None
    if binary is None:
        return None

    module = load_ember_features_module()

    return {
        "sha256": hashlib.sha256(bytez).hexdigest(),
        "histogram": _raw_histogram(bytez),
        "byteentropy": _raw_byteentropy(bytez),
        "strings": _raw_strings(bytez),
        "general": _raw_general(bytez, binary),
        "header": _raw_header(binary),
        "section": _raw_section(binary),
        "imports": _raw_imports(binary),
        "exports": _raw_exports(binary),
        "datadirectories": _raw_datadirectories(binary),
    }


def extract_feature_vector(bytez: bytes) -> Optional[np.ndarray]:
    """Extract the 2381-dim float32 EMBER v2 vector from PE bytes.

    Returns None if the bytes are not a parseable PE.
    """
    raw = extract_raw_features(bytez)
    if raw is None:
        return None
    module = load_ember_features_module()
    extractor = module.PEFeatureExtractor(feature_version=2, print_feature_warning=False)
    vec = extractor.process_raw_features(_normalize_record(raw))
    return vec.astype(np.float32)