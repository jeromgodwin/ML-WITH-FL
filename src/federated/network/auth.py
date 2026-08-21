"""Client authentication — unique identity/credential per endpoint (Phase 19).

Secure registration/provisioning suitable for project scope: each client gets a
unique client_id and a bearer token (HMAC-like). Tokens are stored server-side
in a registry and validated on every request. No shared default credential.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

from fedshield.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class AuthToken:
    client_id: str
    token: str
    role: str  # client | admin
    created_at: float
    expires_at: Optional[float] = None

    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at

    def to_dict(self) -> dict:
        return asdict(self)


class ClientRegistry:
    """Server-side registry of provisioned clients and their credentials.

    Persisted as JSON under <registry_dir>/clients.json (not committed).
    Supports secure provisioning: generate a unique token per client_id.
    """

    def __init__(self, registry_dir: Path | str = "data/server_registry"):
        self.dir = Path(registry_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "clients.json"
        self._clients: Dict[str, AuthToken] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                import json
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for cid, td in data.items():
                    self._clients[cid] = AuthToken(**td)
            except Exception as e:
                logger.warning("failed to load client registry: %s", e)

    def _save(self) -> None:
        import json
        self.path.write_text(json.dumps({k: v.to_dict() for k, v in self._clients.items()}, indent=2), encoding="utf-8")

    def provision_client(self, client_id: str, role: str = "client", ttl_s: Optional[float] = None) -> AuthToken:
        """Provision a new client with a unique token. Raises if already exists."""
        if client_id in self._clients:
            raise ValueError(f"client already provisioned: {client_id}")
        # Generate a strong random token (32 bytes hex)
        token = secrets.token_hex(32)
        now = time.time()
        expires = now + ttl_s if ttl_s else None
        auth = AuthToken(client_id=client_id, token=token, role=role, created_at=now, expires_at=expires)
        self._clients[client_id] = auth
        self._save()
        logger.info("provisioned %s as %s", client_id, role)
        return auth

    def get(self, client_id: str) -> Optional[AuthToken]:
        return self._clients.get(client_id)

    def revoke(self, client_id: str) -> None:
        if client_id in self._clients:
            del self._clients[client_id]
            self._save()

    def authenticate(self, client_id: str, token: str) -> Optional[AuthToken]:
        """Validate client_id + token. Returns AuthToken on success, None on failure."""
        auth = self._clients.get(client_id)
        if auth is None:
            return None
        if auth.is_expired():
            return None
        # Constant-time compare
        if not hmac.compare_digest(auth.token, token):
            return None
        return auth

    def list_clients(self) -> Dict[str, dict]:
        return {k: v.to_dict() for k, v in self._clients.items()}


def provision_client(registry: ClientRegistry, client_id: str, role: str = "client") -> AuthToken:
    return registry.provision_client(client_id, role=role)


def authenticate(registry: ClientRegistry, client_id: str, token: str) -> Optional[AuthToken]:
    return registry.authenticate(client_id, token)
