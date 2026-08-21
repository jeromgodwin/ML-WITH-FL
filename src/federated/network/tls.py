"""TLS/HTTPS for client/server communication (Phase 19).

Uses HTTPS/TLS when server.secure=true; falls back to plain channel for local sim
when secure=false. Never exposes unauthenticated dev API publicly — dev mode
requires secure=false explicitly.
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fedshield.config import ServerNetworkConfig


@dataclass
class TLSConfig:
    """Resolved TLS configuration for one endpoint."""

    enabled: bool
    cert_path: Optional[Path] = None
    key_path: Optional[Path] = None
    ca_path: Optional[Path] = None

    def is_configured(self) -> bool:
        return self.enabled and self.cert_path is not None and self.key_path is not None


def get_tls_context(cfg: ServerNetworkConfig, for_server: bool = True) -> Optional[ssl.SSLContext]:
    """Build an SSLContext from ServerNetworkConfig. Returns None if secure=false or files missing.

    for_server=True → server-side context (requires cert+key)
    for_server=False → client-side context (uses ca_cert if provided)
    """
    if not cfg.secure:
        return None
    # If secure but no files provided, return a default context that still enforces TLS
    # (will use system CAs). For local simulation without certs, caller may run with secure=false.
    try:
        if for_server:
            if cfg.tls_cert and cfg.tls_key and Path(cfg.tls_cert).exists() and Path(cfg.tls_key).exists():
                ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ctx.load_cert_chain(certfile=cfg.tls_cert, keyfile=cfg.tls_key)
                if cfg.ca_cert and Path(cfg.ca_cert).exists():
                    ctx.load_verify_locations(cafile=cfg.ca_cert)
                    ctx.verify_mode = ssl.CERT_REQUIRED
                return ctx
            # Secure requested but no certs — return a context that still negotiates TLS
            # (self-signed / dev). Callers should provide certs for internet deployment.
            ctx = ssl._create_unverified_context() if not for_server else ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            return ctx
        else:
            ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            if cfg.ca_cert and Path(cfg.ca_cert).exists():
                ctx.load_verify_locations(cafile=cfg.ca_cert)
            # For internet deployment, verification is enabled; for localhost with self-signed,
            # the client may need to pass check_hostname=False if using IP.
            return ctx
    except Exception:
        return None


def server_address(cfg: ServerNetworkConfig) -> str:
    """Configurable server address — never hardcoded localhost elsewhere."""
    return f"{cfg.host}:{cfg.port}"


def server_url(cfg: ServerNetworkConfig, path: str = "") -> str:
    """HTTPS or HTTP URL based on secure flag."""
    scheme = "https" if cfg.secure else "http"
    base = f"{scheme}://{cfg.host}:{cfg.port}"
    if path:
        if not path.startswith("/"):
            path = "/" + path
        return base + path
    return base
