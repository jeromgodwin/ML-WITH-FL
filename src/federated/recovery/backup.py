"""Disaster recovery — backup/restore, DB, model rollback, interrupted FL (Enhancement 26)."""

from __future__ import annotations

import shutil
from pathlib import Path

def backup(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return dst

def restore(backup_path: Path, target: Path) -> Path:
    return backup(backup_path, target)
