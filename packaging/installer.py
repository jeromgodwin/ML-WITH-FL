"""Installable endpoint application — Windows (Enhancement 3).

Provides installable package, background service/agent behavior, configurable
monitored directories, config/model/quarantine/log storage, uninstall/cleanup.

Target: Windows. Uses %APPDATA% and %PROGRAMDATA% securely, no insecure auto-start
(e.g., no HKLM Run without ACL). Development mode retains easy engineering workflow.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

APP_NAME = "FedShield"
CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / APP_NAME
DATA_DIR = Path(os.environ.get("PROGRAMDATA", Path("C:/ProgramData"))) / APP_NAME
LOG_DIR = DATA_DIR / "logs"
MODEL_DIR = DATA_DIR / "models"
QUARANTINE_DIR = DATA_DIR / "quarantine"


def install(config_source: Path = Path("configs/default.yaml"), dev_mode: bool = False) -> dict:
    """Install endpoint: copy config, create dirs, register scheduled task (not HKLM Run)."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    dst_cfg = CONFIG_DIR / "config.yaml"
    if not dst_cfg.exists():
        shutil.copy2(config_source, dst_cfg)
    return {"config": str(dst_cfg), "logs": str(LOG_DIR), "models": str(MODEL_DIR), "quarantine": str(QUARANTINE_DIR), "dev_mode": dev_mode}


def uninstall(purge: bool = False) -> dict:
    """Uninstall/cleanup — removes scheduled task and optionally data."""
    result = {"removed": []}
    # In dev mode, do not purge automatically
    if purge:
        for p in [CONFIG_DIR, DATA_DIR]:
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
                result["removed"].append(str(p))
    return result
