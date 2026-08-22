"""Client software version management — compatibility checks (Enhancement 18)."""

from __future__ import annotations

VERSION = "25.0.0"
MIN_SUPPORTED = "20.0.0"

def is_compatible(client_version: str) -> bool:
    return client_version >= MIN_SUPPORTED

def check_compatibility(client_version: str, server_version: str = VERSION) -> dict:
    return {"compatible": is_compatible(client_version), "client": client_version, "server": server_version, "min_supported": MIN_SUPPORTED}
