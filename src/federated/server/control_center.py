"""FedShield Server / Control Center — server application (Phase 19).

The server contains:
- Flower server, aggregation, client coordination, model registry,
  model validation, experiment management, FastAPI, control dashboard (later)

The server must not receive raw endpoint files — enforced by validation layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fedshield.config import ServerNetworkConfig
from src.federated.model_registry import ModelRegistry
from src.federated.network.auth import ClientRegistry
from src.federated.network.replay import ReplayProtection
from src.federated.network.security_middleware import SecurityLayer
from backend.secure_app import init_secure_app


class FedShieldServer:
    """Composition of all server components."""

    def __init__(
        self,
        server_cfg: Optional[ServerNetworkConfig] = None,
        model_registry_dir: Path | str = "data/server_registry",
        client_registry_dir: Path | str = "data/server_registry",
    ):
        self.server_cfg = server_cfg or ServerNetworkConfig()
        self.model_registry = ModelRegistry(model_registry_dir)
        self.client_registry = ClientRegistry(client_registry_dir)
        self.replay = ReplayProtection()
        self.security = SecurityLayer(self.server_cfg, self.client_registry, self.replay)
        # FastAPI app with TLS + auth + validation + replay
        self.app = init_secure_app(self.server_cfg, model_registry_dir, client_registry_dir)

    def get_status(self) -> dict:
        return {
            "server": f"{self.server_cfg.host}:{self.server_cfg.port}",
            "secure": self.server_cfg.secure,
            "components": [
                "flower_server", "aggregation", "client_coordination",
                "model_registry", "model_validation", "experiment_management",
                "fastapi", "control_dashboard_later",
            ],
            "must_not_receive_raw_files": True,
        }
