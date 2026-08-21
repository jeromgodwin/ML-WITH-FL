"""FedShield Server / Control Center — secure FastAPI (Phase 19).

Contains: Flower server, aggregation, client coordination, model registry,
model validation, experiment management, FastAPI, control dashboard (later).

Security layer (TLS + auth + validation + replay) sits between client and server.
The server must not receive raw endpoint files — enforced in validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from fedshield.config import ServerNetworkConfig
from src.federated.model_registry import ModelRegistry
from src.federated.network.auth import ClientRegistry
from src.federated.network.replay import ReplayProtection
from src.federated.network.security_middleware import SecurityLayer
from src.federated.network.validation import MessageValidationError, validate_message

# Global singletons (configured via init_secure_app)
_security: Optional[SecurityLayer] = None
_client_registry: Optional[ClientRegistry] = None
_replay: Optional[ReplayProtection] = None
_model_registry: Optional[ModelRegistry] = None


def init_secure_app(
    server_cfg: ServerNetworkConfig,
    model_registry_dir: Path | str = "data/server_registry",
    client_registry_dir: Path | str = "data/server_registry",
) -> FastAPI:
    global _security, _client_registry, _replay, _model_registry
    _client_registry = ClientRegistry(client_registry_dir)
    _replay = ReplayProtection()
    _security = SecurityLayer(server_cfg, _client_registry, _replay)
    _model_registry = ModelRegistry(model_registry_dir)

    app = FastAPI(
        title="FedShield Control Center",
        docs_url="/docs" if not server_cfg.secure else None,  # do not expose unauthenticated dev API publicly when secure
        redoc_url=None if server_cfg.secure else "/redoc",
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "secure": server_cfg.secure, "host": server_cfg.host, "port": server_cfg.port}

    @app.post("/api/v1/register")
    def register_client(payload: Dict[str, Any], x_client_id: str = Header(None), authorization: str = Header(None)):
        # Registration requires admin token (provisioning)
        ctx = _security.authenticate(x_client_id or "", (authorization or "").replace("Bearer ", "")) if _security else None
        if not _security or not _security.authorize(ctx, "provision_client"):
            raise HTTPException(status_code=403, detail="admin required for registration")
        client_id = payload.get("client_id")
        if not client_id:
            raise HTTPException(status_code=400, detail="missing client_id")
        try:
            auth = _client_registry.provision_client(client_id, role=payload.get("role", "client"))
            return {"client_id": auth.client_id, "role": auth.role}
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.post("/api/v1/fl/update")
    def submit_update(
        payload: Dict[str, Any],
        x_client_id: str = Header(None),
        authorization: str = Header(None),
        x_request_id: Optional[str] = Header(None),
    ):
        # Authenticate
        token = (authorization or "").replace("Bearer ", "")
        ctx = _security.authenticate(x_client_id or "", token) if _security else None
        if ctx is None:
            raise HTTPException(status_code=401, detail="invalid client identity")

        # Validate message (schema, identity, metadata, round, version, format)
        try:
            # Inject request_id from header if not in payload
            if x_request_id and "request_id" not in payload:
                payload["request_id"] = x_request_id
            if "client_id" not in payload:
                payload["client_id"] = x_client_id
            _security.validate_and_check_replay(payload, ctx, operation="submit_update")
        except MessageValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))

        # Server must not receive raw endpoint files — reject if present
        if any(k in payload for k in ("raw_file", "file_bytes", "pe_bytes", "raw_bytes")):
            raise HTTPException(status_code=400, detail="server must not receive raw endpoint files")

        return {"status": "accepted", "client_id": ctx.client_id, "round": payload.get("round")}

    @app.get("/api/v1/model/active")
    def get_active_model(x_client_id: str = Header(None), authorization: str = Header(None)):
        token = (authorization or "").replace("Bearer ", "")
        ctx = _security.authenticate(x_client_id or "", token) if _security else None
        if ctx is None:
            raise HTTPException(status_code=401, detail="invalid client identity")
        if not _security.authorize(ctx, "fetch_model"):
            raise HTTPException(status_code=403, detail="client not authorized to fetch model")
        active = _model_registry.get_active() if _model_registry else None
        if active is None:
            return JSONResponse(status_code=404, content={"detail": "no active model"})
        return active.to_dict()

    @app.get("/api/v1/admin/models")
    def list_models_admin(x_client_id: str = Header(None), authorization: str = Header(None)):
        token = (authorization or "").replace("Bearer ", "")
        ctx = _security.authenticate(x_client_id or "", token) if _security else None
        if not _security or not _security.authorize(ctx, "list_all_models_admin"):
            raise HTTPException(status_code=403, detail="admin required")
        return [e.to_dict() for e in (_model_registry.list_all() if _model_registry else [])]

    return app


def get_security_layer() -> Optional[SecurityLayer]:
    return _security
