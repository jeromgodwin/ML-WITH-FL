"""System service — no ML/FL logic in handlers (Phase 20 §1).

Provides: health, server status, protection/client status, active model, model version
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fedshield.config import ServerNetworkConfig
from src.federated.model_registry import ModelRegistry


class SystemService:
    def __init__(self, server_cfg: ServerNetworkConfig, model_registry: Optional[ModelRegistry] = None):
        self.server_cfg = server_cfg
        self.model_registry = model_registry or ModelRegistry("data/server_registry")

    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "secure": self.server_cfg.secure, "host": self.server_cfg.host, "port": self.server_cfg.port}

    def server_status(self) -> Dict[str, Any]:
        return {
            "server": f"{self.server_cfg.host}:{self.server_cfg.port}",
            "secure": self.server_cfg.secure,
            "tls_configured": bool(self.server_cfg.tls_cert and self.server_cfg.tls_key),
        }

    def protection_status(self) -> Dict[str, Any]:
        active = None
        version = None
        try:
            a = self.model_registry.get_active()
            if a:
                active = a.to_dict()
                version = a.version
        except Exception:
            pass
        return {"protection": "active" if active else "no_active_model", "active_model": version, "model": active}

    def active_model(self) -> Optional[Dict[str, Any]]:
        try:
            a = self.model_registry.get_active()
            return a.to_dict() if a else None
        except Exception:
            return None

    def model_version(self) -> Optional[str]:
        try:
            a = self.model_registry.get_active()
            return a.version if a else None
        except Exception:
            return None
