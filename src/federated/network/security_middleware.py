"""Security layer between FedShield Client Agent and Server (Phase 19).

The network-security functionality is a security layer between the two main
applications, not a third standalone application.

This middleware enforces:
- TLS (via TLSConfig)
- Client authentication (bearer token)
- Authorization (client vs admin operations)
- Message/update validation
- Replay protection

It is used by the FastAPI server as dependency injection and by the Flower
gRPC interceptor (or wrapper) for FL updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from fedshield.config import ServerNetworkConfig, ClientIdentityConfig
from src.federated.network.auth import ClientRegistry
from src.federated.network.replay import ReplayProtection
from src.federated.network.tls import TLSConfig, get_tls_context
from src.federated.network.validation import MessageValidationError, validate_message


# Operations that require admin role
ADMIN_OPERATIONS: Set[str] = {
    "approve_model",
    "reject_model",
    "rollback_model",
    "list_all_models_admin",
    "experiment_management",
    "provision_client",
    "revoke_client",
}

# Operations allowed for client role
CLIENT_OPERATIONS: Set[str] = {
    "submit_update",
    "fetch_model",
    "get_active_model",
    "register",
    "heartbeat",
    "get_status",
}


@dataclass
class SecurityContext:
    client_id: str
    role: str
    authenticated: bool
    operation: Optional[str] = None


class SecurityLayer:
    """Composes TLS, auth, authz, validation, replay into one layer."""

    def __init__(
        self,
        server_cfg: ServerNetworkConfig,
        client_registry: Optional[ClientRegistry] = None,
        replay_protection: Optional[ReplayProtection] = None,
    ):
        self.server_cfg = server_cfg
        self.tls = TLSConfig(
            enabled=server_cfg.secure,
            cert_path=server_cfg.tls_cert,
            key_path=server_cfg.tls_key,
            ca_path=server_cfg.ca_cert,
        ) if server_cfg else TLSConfig(enabled=False)
        self.client_registry = client_registry or ClientRegistry()
        self.replay = replay_protection or ReplayProtection()

    def authenticate(self, client_id: str, token: Optional[str]) -> Optional[SecurityContext]:
        if not client_id or not token:
            return None
        auth = self.client_registry.authenticate(client_id, token)
        if auth is None:
            return None
        return SecurityContext(client_id=client_id, role=auth.role, authenticated=True)

    def authorize(self, ctx: Optional[SecurityContext], operation: str) -> bool:
        if ctx is None or not ctx.authenticated:
            return False
        if operation in ADMIN_OPERATIONS:
            return ctx.role == "admin"
        if operation in CLIENT_OPERATIONS:
            return ctx.role in ("client", "admin")
        # Unknown operations default to admin-only (safe)
        return ctx.role == "admin"

    def validate_and_check_replay(
        self,
        payload: Dict[str, Any],
        ctx: Optional[SecurityContext],
        operation: Optional[str] = None,
    ) -> Dict[str, Any]:
        # 1. Validate schema/identity/metadata/round/version/format
        validate_message(payload, expected_client_id=ctx.client_id if ctx else None)

        # 2. Replay protection
        request_id = payload.get("request_id") or payload.get("requestId")
        round_num = payload.get("round") or payload.get("round_number")
        model_ver = payload.get("model_version") or payload.get("version")
        if ctx:
            accepted, reason = self.replay.check_and_record(
                client_id=ctx.client_id,
                request_id=str(request_id) if request_id else None,
                round_number=int(round_num) if round_num is not None else None,
                model_version=str(model_ver) if model_ver else None,
            )
            if not accepted:
                raise MessageValidationError(f"replay/stale rejected: {reason}")

        # 3. Authorization (if operation given)
        if operation and not self.authorize(ctx, operation):
            raise PermissionError(f"unauthorized: {ctx.role if ctx else 'unauthenticated'} cannot perform {operation}")

        return payload

    def server_url(self, path: str = "") -> str:
        from src.federated.network.tls import server_url
        return server_url(self.server_cfg, path)

    def is_tls_enabled(self) -> bool:
        return bool(self.server_cfg.secure)
