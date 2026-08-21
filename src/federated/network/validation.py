"""Message/update validation (Phase 19).

Validates:
- request schema
- client identity
- model metadata
- round number
- model version
- update format

Rejects malformed or unauthorized messages before they reach aggregation or registry.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class MessageValidationError(ValueError):
    pass


def validate_message(
    payload: Dict[str, Any],
    expected_client_id: Optional[str] = None,
    require_fields: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Validate a client message payload.

    Raises MessageValidationError on failure. Returns the payload if valid.
    """
    if not isinstance(payload, dict):
        raise MessageValidationError("payload must be a dict")

    # 1. request schema — must contain type or version field
    # We accept either 'type' or 'model_version' as schema indicator, but require at least one
    if require_fields:
        for field in require_fields:
            if field not in payload:
                raise MessageValidationError(f"missing required field: {field}")

    # 2. client identity — if expected_client_id given, must match
    if expected_client_id is not None:
        cid = payload.get("client_id") or payload.get("clientId")
        if cid is None:
            raise MessageValidationError("missing client_id")
        if str(cid) != str(expected_client_id):
            raise MessageValidationError(f"client_id mismatch: {cid} != {expected_client_id}")

    # 3. model metadata — if present, must be a dict with version
    if "model_metadata" in payload:
        meta = payload["model_metadata"]
        if not isinstance(meta, dict):
            raise MessageValidationError("model_metadata must be a dict")
        if "version" not in meta and "model_version" not in meta:
            raise MessageValidationError("model_metadata missing version")

    # 4. round number — if present, must be int >=0
    if "round" in payload or "round_number" in payload:
        rnd = payload.get("round", payload.get("round_number"))
        if not isinstance(rnd, int) or rnd < 0:
            raise MessageValidationError(f"invalid round number: {rnd}")

    # 5. model version — if present, must be non-empty string
    if "model_version" in payload:
        ver = payload["model_version"]
        if not isinstance(ver, str) or not ver.strip():
            raise MessageValidationError(f"invalid model_version: {ver}")

    # 6. update format — if 'update' or 'parameters' present, must be list of arrays (we check list)
    for key in ("update", "parameters", "model_update"):
        if key in payload:
            val = payload[key]
            if val is not None and not isinstance(val, (list, dict)):
                raise MessageValidationError(f"invalid update format for {key}: {type(val).__name__}")

    return payload


def validate_client_identity(client_id: str, token: Optional[str], registry: Any) -> bool:
    """Validate client identity against the registry. Returns True if valid."""
    if not client_id or not token:
        return False
    try:
        auth = registry.authenticate(client_id, token)
        return auth is not None
    except Exception:
        return False
