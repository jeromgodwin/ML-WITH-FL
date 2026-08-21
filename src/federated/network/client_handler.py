"""Client-side network handling — endpoint remains operational without server (Phase 19).

When server unavailable:
- endpoint detection continues (local inference)
- current model remains active (local registry cache)
- FL training is deferred
- queued/unsent federated work is handled safely (persisted queue, retry)

The client never sends raw endpoint files to the server — only model updates
and scalar metrics (enforced by validation layer).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fedshield.logging_setup import get_logger

logger = get_logger(__name__)


class NetworkFailureHandler:
    """Handles server unavailability for the endpoint.

    The endpoint must remain operational without the server. This handler
    provides: local fallback, queuing, and safe retry.
    """

    def __init__(self, queue_dir: Path | str = "data/client_queue"):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._offline_since: Optional[float] = None

    def is_server_available(self, check_fn) -> bool:
        """Probe server availability via a callable (e.g., health check)."""
        try:
            check_fn()
            self._offline_since = None
            return True
        except Exception as e:
            if self._offline_since is None:
                self._offline_since = time.time()
                logger.warning("server unavailable — endpoint continues with current model: %s", e)
            return False

    def enqueue(self, payload: Dict[str, Any]) -> Path:
        """Persist unsent federated work safely (never raw files)."""
        # Safety: refuse to queue raw file bytes (defense in depth)
        if any(k in payload for k in ("raw_file", "file_bytes", "pe_bytes")):
            raise ValueError("refusing to queue raw endpoint file — server must not receive raw files")
        ts = int(time.time() * 1000)
        p = self.queue_dir / f"queued_{ts}_{payload.get('request_id','')}.json"
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("queued federated work %s (offline)", p.name)
        return p

    def drain_queue(self, send_fn) -> int:
        """Attempt to send queued items when server returns. Returns count sent."""
        sent = 0
        for p in sorted(self.queue_dir.glob("queued_*.json")):
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                send_fn(payload)
                p.unlink()
                sent += 1
            except Exception as e:
                logger.warning("failed to send queued %s: %s", p.name, e)
                break  # stop on first failure (still offline)
        if sent:
            logger.info("drained %d queued items", sent)
        return sent

    def handle_offline_detection(self) -> Dict[str, Any]:
        """Endpoint detection continues using current active model (local cache)."""
        return {
            "status": "offline_mode",
            "detection": "active",
            "model": "current_active_cached",
            "fl_training": "deferred",
            "message": "endpoint operational without server — using local model cache",
        }


class EndpointClientApp:
    """FedShield Client Agent composition (Phase 19 §1).

    Contains: file monitor, PE/static analysis, feature extraction, local
    inference, risk engine, quarantine, notifications, local history, resource
    monitor, drift detector, FL client, local model registry/cache.
    """

    def __init__(
        self,
        monitor: Any = None,
        detector: Any = None,
        registry: Any = None,
        history: Any = None,
        quarantine: Any = None,
        resource_monitor: Any = None,
        drift_detector: Any = None,
        network_handler: Optional[NetworkFailureHandler] = None,
    ):
        self.monitor = monitor
        self.detector = detector
        self.registry = registry
        self.history = history
        self.quarantine = quarantine
        self.resource_monitor = resource_monitor
        self.drift_detector = drift_detector
        self.network_handler = network_handler or NetworkFailureHandler()
        # Local model cache — always available even offline
        self._active_model_version: Optional[str] = None
        if registry is not None:
            try:
                active = registry.get_active()
                if active:
                    self._active_model_version = active.version
            except Exception:
                pass

    def get_active_model_version(self) -> Optional[str]:
        return self._active_model_version

    def is_operational_without_server(self) -> bool:
        """Endpoint must remain operational without the server."""
        # Check local inference still works (model cached)
        if self.registry is None:
            return True  # no registry yet, but detection can still run if model exists
        try:
            active = self.registry.get_active()
            return active is not None
        except Exception:
            # Even if registry unreadable, endpoint continues with cached version
            return self._active_model_version is not None
