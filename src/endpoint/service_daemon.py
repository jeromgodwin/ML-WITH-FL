"""Background service/agent behavior — runs without terminal (Enhancement 3)."""

from __future__ import annotations

import time
from pathlib import Path

from src.endpoint.client_agent import FedShieldClientAgent


def run_service(registry_dir: Path | str = "data/client_registry", poll_interval: float = 5.0):
    """Run as background service (no terminal required). In dev mode, use python src/endpoint/service_daemon.py --dev."""
    agent = FedShieldClientAgent(registry_dir=registry_dir)
    agent.startup()
    try:
        while agent.state.value != "STOPPED":
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        agent.shutdown()
