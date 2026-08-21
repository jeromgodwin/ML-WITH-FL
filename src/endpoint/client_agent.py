"""FedShield Client Agent — endpoint application (Phase 19).

The client contains 11 components and remains operational without the server:

- file monitor, PE/static analysis, feature extraction, local inference,
  risk engine, quarantine, notifications, local history, resource monitor,
  drift detector, FL client, local model registry/cache

If the server is unavailable, detection continues with the current active model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from src.endpoint.monitor import FileMonitor
from src.endpoint.detector import Detector
from src.federated.model_registry import ModelRegistry
from src.federated.network.client_handler import EndpointClientApp, NetworkFailureHandler
from src.endpoint.history import HistoryStore
from src.endpoint.quarantine import QuarantineManager
from src.endpoint.resource.monitor import ResourceMonitor
from src.drift.detector import DriftDetector


class FedShieldClientAgent:
    """Composition of all endpoint components — remains operational offline."""

    def __init__(
        self,
        registry_dir: Path | str = "data/client_registry",
        queue_dir: Path | str = "data/client_queue",
        monitor: Optional[FileMonitor] = None,
        history: Optional[HistoryStore] = None,
    ):
        self.registry = ModelRegistry(registry_dir)
        self.network_handler = NetworkFailureHandler(queue_dir=queue_dir)
        # Local model cache — always available
        self._active_version: Optional[str] = None
        try:
            active = self.registry.get_active()
            if active:
                self._active_version = active.version
        except Exception:
            pass

        # Placeholders for the 11 components (wired via dependency injection)
        self.monitor = monitor
        self.history = history
        self.resource_monitor = ResourceMonitor()
        self.drift_detector = DriftDetector()
        self.client_app = EndpointClientApp(
            monitor=monitor,
            history=history,
            registry=self.registry,
            network_handler=self.network_handler,
            resource_monitor=self.resource_monitor,
            drift_detector=self.drift_detector,
        )

    def get_status(self) -> dict:
        return {
            "active_model": self._active_version or self.client_app.get_active_model_version(),
            "operational_without_server": self.client_app.is_operational_without_server(),
            "components": [
                "file_monitor", "pe_static_analysis", "feature_extraction",
                "local_inference", "risk_engine", "quarantine", "notifications",
                "local_history", "resource_monitor", "drift_detector",
                "fl_client", "local_model_registry_cache",
            ],
        }

    def handle_server_unavailable(self) -> dict:
        """When server unavailable: detection continues, model remains active, FL deferred."""
        return self.network_handler.handle_offline_detection()
