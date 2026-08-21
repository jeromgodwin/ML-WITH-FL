"""Automatic endpoint model updater (Phase 18).

A client must be able to discover a new approved model, verify it, and
activate it without manually replacing model files.
If the network/server is unavailable: continue using the current active model.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from fedshield.logging_setup import get_logger
from src.federated.model_registry import ModelRegistry
from src.federated.registry.deployment import endpoint_discover_and_update

logger = get_logger(__name__)


class EndpointModelUpdater:
    """Polls a registry for new ACTIVE models and updates the endpoint automatically."""

    def __init__(
        self,
        registry: ModelRegistry | Path | str,
        endpoint_model_dir: Path | str,
        poll_interval_s: float = 10.0,
    ):
        if isinstance(registry, (str, Path)):
            registry = ModelRegistry(registry)
        self.registry = registry
        self.endpoint_model_dir = Path(endpoint_model_dir)
        self.poll_interval_s = poll_interval_s
        self._current_version: Optional[str] = None
        # Seed from endpoint dir if already has active_version.txt
        try:
            vfile = self.endpoint_model_dir / "active_version.txt"
            if vfile.exists():
                self._current_version = vfile.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    @property
    def current_version(self) -> Optional[str]:
        return self._current_version

    def check_and_update(self) -> Optional[str]:
        """One poll cycle: discover new ACTIVE, verify, deploy. Returns new version or None.

        If the registry is unavailable, returns None and keeps current (graceful).
        """
        try:
            entry = endpoint_discover_and_update(
                self.registry, self.endpoint_model_dir, self._current_version
            )
        except Exception as e:
            logger.warning("updater check failed (offline? %s) — keeping %s", e, self._current_version)
            return None

        if entry is not None:
            self._current_version = entry.version
            return entry.version
        return None

    def run_forever(self, stop_after: Optional[int] = None) -> None:
        """Blocking poll loop (for service integration)."""
        count = 0
        while True:
            self.check_and_update()
            count += 1
            if stop_after is not None and count >= stop_after:
                break
            time.sleep(self.poll_interval_s)
