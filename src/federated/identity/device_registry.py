"""Client identity and device registration — unique identity, rotation, revocation (Enhancement 12)."""

from __future__ import annotations

from src.federated.network.auth import ClientRegistry

class DeviceRegistry(ClientRegistry):
    """Extends ClientRegistry with rotation and disabled state."""
    def rotate(self, client_id: str) -> str:
        auth = self.get(client_id)
        if not auth:
            raise KeyError(client_id)
        import secrets
        new_token = secrets.token_hex(32)
        auth.token = new_token
        self._save()
        return new_token
    def disable(self, client_id: str) -> None:
        auth = self.get(client_id)
        if auth:
            auth.role = "disabled"
            self._save()
