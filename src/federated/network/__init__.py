"""Secure client/server networking — security layer between Client Agent and Server (Phase 19).

The network-security layer is not a third app; it is middleware that secures
the channel between:
  FedShield Client Agent (endpoint) <-> FedShield Server / Control Center
"""

from src.federated.network.tls import TLSConfig, get_tls_context
from src.federated.network.auth import ClientRegistry, AuthToken, provision_client, authenticate
from src.federated.network.validation import validate_message, MessageValidationError
from src.federated.network.replay import ReplayProtection
from src.federated.network.security_middleware import SecurityLayer

__all__ = [
    "TLSConfig",
    "get_tls_context",
    "ClientRegistry",
    "AuthToken",
    "provision_client",
    "authenticate",
    "validate_message",
    "MessageValidationError",
    "ReplayProtection",
    "SecurityLayer",
]
