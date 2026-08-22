"""Model integrity and distribution security — hash, verify, atomic activate (Enhancement 11)."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_and_activate(src: Path, dst: Path, expected_hash: str) -> bool:
    """Verify hash, then atomically replace dst. Reject if fails, keep previous."""
    if not src.exists():
        return False
    if hash_file(src) != expected_hash:
        return False
    # Atomic: copy to temp then rename
    tmp = dst.with_suffix(".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)
    return True
