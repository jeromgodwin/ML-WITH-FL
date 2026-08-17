"""Static PE file analysis: format detection, metadata extraction, safety checks.

This module performs safe static analysis only - never executes files.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fedshield.logging_setup import get_logger

logger = get_logger(__name__)

# PE format constants
IMAGE_DOS_SIGNATURE = b"MZ"
IMAGE_NT_SIGNATURE = b"PE\x00\x00"

# Machine types
IMAGE_FILE_MACHINE_I386 = 0x014c
IMAGE_FILE_MACHINE_AMD64 = 0x8664

# Characteristics
IMAGE_FILE_EXECUTABLE_IMAGE = 0x0002
IMAGE_FILE_DLL = 0x2000
IMAGE_FILE_SYSTEM = 0x1000


@dataclass
class PEInfo:
    """Extracted PE file information."""

    is_pe: bool
    is_64bit: bool
    is_dll: bool
    is_sys: bool
    is_executable: bool
    machine: int
    num_sections: int
    entry_point: int
    image_base: int
    section_names: list[str]
    section_entropies: list[float]
    imports: list[str]
    exports: list[str]
    size_of_code: int
    size_of_initialized_data: int
    size_of_uninitialized_data: int
    checksum: int
    timestamp: int
    # Raw metadata
    file_size: int
    sha256: str


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def calculate_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of byte data."""
    if not data:
        return 0.0
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    entropy = 0.0
    n = len(data)
    for count in freq:
        if count:
            p = count / n
            entropy -= p * math.log2(p)
    return entropy


def read_dos_header(f) -> Optional[tuple[int, int]]:
    """Read DOS header, return (e_magic, e_lfanew) or None."""
    data = f.read(64)
    if len(data) < 64:
        return None
    e_magic, _, _, _, _, _, _, _, _, _, _, _, _, _, _, e_lfanew = struct.unpack(
        "<HHHHHHHHHHHHHHHHI", data
    )
    return e_magic, e_lfanew


def read_pe_header(f, offset: int) -> Optional[dict]:
    """Read PE (NT) headers at given offset."""
    f.seek(offset)
    sig = f.read(4)
    if sig != IMAGE_NT_SIGNATURE:
        return None

    # COFF File Header (20 bytes)
    coff = f.read(20)
    if len(coff) < 20:
        return None
    (
        machine,
        num_sections,
        timestamp,
        _ptr_to_sym_table,
        _num_symbols,
        size_of_opt_header,
        characteristics,
    ) = struct.unpack("<HHIIIHH", coff[:20])

    # Optional Header
    opt_header = f.read(size_of_opt_header)
    if len(opt_header) < size_of_opt_header:
        return None

    # Determine 32 vs 64 bit by magic
    magic = struct.unpack("<H", opt_header[:2])[0]
    is_64bit = magic == 0x20b  # PE32+

    if is_64bit:
        # PE32+ optional header (112 bytes for standard fields)
        fmt = "<HBBIIIIQQQQQQQQQQQQ"
        vals = struct.unpack(fmt, opt_header[:112])
        (
            _magic,
            _major_linker_version,
            _minor_linker_version,
            size_of_code,
            size_of_initialized_data,
            size_of_uninitialized_data,
            entry_point,
            base_of_code,
            image_base,
            _section_alignment,
            _file_alignment,
            _major_os_version,
            _minor_os_version,
            _major_image_version,
            _minor_image_version,
            _major_subsystem_version,
            _minor_subsystem_version,
            _win32_version_value,
            _size_of_image,
            _size_of_headers,
            checksum,
        ) = vals
    else:
        # PE32 optional header (96 bytes for standard fields)
        fmt = "<HBBIIIIIIQQQQQQQQQQQ"
        vals = struct.unpack(fmt, opt_header[:96])
        (
            _magic,
            _major_linker_version,
            _minor_linker_version,
            size_of_code,
            size_of_initialized_data,
            size_of_uninitialized_data,
            entry_point,
            base_of_code,
            _base_of_data,
            image_base,
            _section_alignment,
            _file_alignment,
            _major_os_version,
            _minor_os_version,
            _major_image_version,
            _minor_image_version,
            _major_subsystem_version,
            _minor_subsystem_version,
            _win32_version_value,
            _size_of_image,
            _size_of_headers,
            checksum,
        ) = vals

    # Section headers follow optional header
    section_offset = offset + 24 + size_of_opt_header
    f.seek(section_offset)

    section_names = []
    section_entropies = []

    for _ in range(num_sections):
        sec_data = f.read(40)
        if len(sec_data) < 40:
            break
        name = sec_data[:8].rstrip(b"\x00").decode("ascii", errors="ignore")
        (
            _virtual_size,
            _virtual_address,
            size_of_raw_data,
            ptr_to_raw_data,
            _ptr_to_relocations,
            _ptr_to_linenumbers,
            _num_relocations,
            _num_linenumbers,
            _characteristics,
        ) = struct.unpack("<IIIIIIHHI", sec_data[8:40])

        section_names.append(name)

        # Read section data for entropy
        if size_of_raw_data > 0 and ptr_to_raw_data > 0:
            cur_pos = f.tell()
            f.seek(ptr_to_raw_data)
            sec_content = f.read(min(size_of_raw_data, 65536))  # Cap at 64KB for entropy
            f.seek(cur_pos)
            section_entropies.append(calculate_entropy(sec_content))
        else:
            section_entropies.append(0.0)

    # Imports/Exports - simplified extraction (would need full PE parsing for complete)
    # For now, return empty - feature extraction will handle this
    imports = []
    exports = []

    return {
        "machine": machine,
        "num_sections": num_sections,
        "timestamp": timestamp,
        "characteristics": characteristics,
        "size_of_code": size_of_code,
        "size_of_initialized_data": size_of_initialized_data,
        "size_of_uninitialized_data": size_of_uninitialized_data,
        "entry_point": entry_point,
        "image_base": image_base,
        "checksum": checksum,
        "is_64bit": is_64bit,
        "is_dll": bool(characteristics & IMAGE_FILE_DLL),
        "is_sys": bool(characteristics & IMAGE_FILE_SYSTEM),
        "is_executable": bool(characteristics & IMAGE_FILE_EXECUTABLE_IMAGE),
        "section_names": section_names,
        "section_entropies": section_entropies,
        "imports": imports,
        "exports": exports,
    }


def analyze_pe_file(path: Path) -> PEInfo:
    """Perform static analysis on a PE file.

    Returns PEInfo with extracted metadata. Never executes the file.
    """
    file_size = path.stat().st_size
    sha256 = compute_sha256(path)

    with open(path, "rb") as f:
        # Read DOS header
        dos = read_dos_header(f)
        if not dos:
            return PEInfo(
                is_pe=False,
                is_64bit=False,
                is_dll=False,
                is_sys=False,
                is_executable=False,
                machine=0,
                num_sections=0,
                entry_point=0,
                image_base=0,
                section_names=[],
                section_entropies=[],
                imports=[],
                exports=[],
                size_of_code=0,
                size_of_initialized_data=0,
                size_of_uninitialized_data=0,
                checksum=0,
                timestamp=0,
                file_size=file_size,
                sha256=sha256,
            )

        e_magic, e_lfanew = dos
        if e_magic != 0x5a4d:  # "MZ" little-endian
            return PEInfo(
                is_pe=False,
                is_64bit=False,
                is_dll=False,
                is_sys=False,
                is_executable=False,
                machine=0,
                num_sections=0,
                entry_point=0,
                image_base=0,
                section_names=[],
                section_entropies=[],
                imports=[],
                exports=[],
                size_of_code=0,
                size_of_initialized_data=0,
                size_of_uninitialized_data=0,
                checksum=0,
                timestamp=0,
                file_size=file_size,
                sha256=sha256,
            )

        # Read PE header
        pe = read_pe_header(f, e_lfanew)
        if not pe:
            return PEInfo(
                is_pe=False,
                is_64bit=False,
                is_dll=False,
                is_sys=False,
                is_executable=False,
                machine=0,
                num_sections=0,
                entry_point=0,
                image_base=0,
                section_names=[],
                section_entropies=[],
                imports=[],
                exports=[],
                size_of_code=0,
                size_of_initialized_data=0,
                size_of_uninitialized_data=0,
                checksum=0,
                timestamp=0,
                file_size=file_size,
                sha256=sha256,
            )

    return PEInfo(
        is_pe=True,
        is_64bit=pe["is_64bit"],
        is_dll=pe["is_dll"],
        is_sys=pe["is_sys"],
        is_executable=pe["is_executable"],
        machine=pe["machine"],
        num_sections=pe["num_sections"],
        entry_point=pe["entry_point"],
        image_base=pe["image_base"],
        section_names=pe["section_names"],
        section_entropies=pe["section_entropies"],
        imports=pe["imports"],
        exports=pe["exports"],
        size_of_code=pe["size_of_code"],
        size_of_initialized_data=pe["size_of_initialized_data"],
        size_of_uninitialized_data=pe["size_of_uninitialized_data"],
        checksum=pe["checksum"],
        timestamp=pe["timestamp"],
        file_size=file_size,
        sha256=sha256,
    )


def classify_file_type(pe_info: PEInfo, extension: str) -> str:
    """Classify file type based on PE info and extension."""
    if not pe_info.is_pe:
        return "unknown"

    ext = extension.lower()
    if pe_info.is_dll or ext == ".dll":
        return "pe_dll"
    if pe_info.is_sys or ext == ".sys":
        return "pe_sys"
    if pe_info.is_executable or ext in (".exe", ".scr", ".com"):
        return "pe_exe"
    return "pe_unknown"
