"""FedShield Control Center — FastAPI backend with service abstractions (Phase 20).

Route handlers contain no core ML/FL logic — they delegate to services.
Security: authentication, authorization, request validation, safe error handling,
rate limiting, no arbitrary filesystem access.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from fedshield.config import ServerNetworkConfig
from src.federated.model_registry import ModelRegistry
from src.federated.network.auth import ClientRegistry
from src.federated.network.replay import ReplayProtection
from src.federated.network.security_middleware import SecurityLayer

from backend.services.system_service import SystemService
from backend.services.client_service import ClientService
from backend.services.detection_service import DetectionService
from backend.services.fl_service import FLService
from backend.services.resource_service import ResourceService
from backend.services.drift_service import DriftService
from backend.services.model_service import ModelService
from backend.security.rate_limiter import RateLimiter
from backend.security.safe_fs import safe_path

# Singletons configured via create_app
_security: Optional[SecurityLayer] = None
_rate_limiter = RateLimiter(max_requests=120, window_s=60.0)


def create_app(
    server_cfg: Optional[ServerNetworkConfig] = None,
    model_registry_dir: Path | str = "data/server_registry",
    client_registry_dir: Path | str = "data/server_registry",
) -> FastAPI:
    global _security
    server_cfg = server_cfg or ServerNetworkConfig()
    client_registry = ClientRegistry(client_registry_dir)
    model_registry = ModelRegistry(model_registry_dir)
    replay = ReplayProtection()
    _security = SecurityLayer(server_cfg, client_registry, replay)

    # Services (abstractions)
    system_svc = SystemService(server_cfg, model_registry)
    client_svc = ClientService(client_registry, model_registry)
    detection_svc = DetectionService()
    fl_svc = FLService()
    resource_svc = ResourceService()
    drift_svc = DriftService()
    model_svc = ModelService(model_registry)

    app = FastAPI(
        title="FedShield Control Center",
        version="20.0.0",
        docs_url="/docs" if not server_cfg.secure else None,
        redoc_url=None if server_cfg.secure else "/redoc",
    )

    # --- helpers ---
    def _auth(x_client_id: Optional[str], authorization: Optional[str]):
        token = (authorization or "").replace("Bearer ", "")
        ctx = _security.authenticate(x_client_id or "", token) if _security else None
        if ctx is None:
            raise HTTPException(status_code=401, detail="invalid client identity")
        return ctx

    def _rate_limit(request: Request):
        key = request.client.host if request.client else "unknown"
        if not _rate_limiter.is_allowed(key):
            raise HTTPException(status_code=429, detail="rate limit exceeded")

    @app.exception_handler(Exception)
    async def safe_error_handler(request: Request, exc: Exception):
        # Safe error handling — do not leak internals
        if isinstance(exc, HTTPException):
            raise exc
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    # 1. SYSTEM
    @app.get("/health")
    def health():
        return system_svc.health()

    @app.get("/api/v1/status")
    def server_status():
        return system_svc.server_status()

    @app.get("/api/v1/protection/status")
    def protection_status():
        return system_svc.protection_status()

    @app.get("/api/v1/model/active")
    def get_active_model(x_client_id: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
        ctx = _auth(x_client_id, authorization)
        # Any authenticated client can fetch active model
        active = system_svc.active_model()
        if not active:
            raise HTTPException(status_code=404, detail="no active model")
        return active

    @app.get("/api/v1/model/version")
    def model_version():
        ver = system_svc.model_version()
        if not ver:
            raise HTTPException(status_code=404, detail="no active model")
        return {"version": ver}

    # 2. CLIENT MANAGEMENT
    @app.get("/api/v1/clients")
    def list_clients(x_client_id: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
        ctx = _auth(x_client_id, authorization)
        # Admin or client can list (but sensitive data filtered by service)
        return client_svc.list_clients()

    @app.get("/api/v1/clients/{client_id}")
    def get_client(client_id: str, x_client_id: Optional[str] = Header(None), authorization: Optional[str] = Header(None), request: Request = None):
        _rate_limit(request)
        ctx = _auth(x_client_id, authorization)
        # Validate client_id format (no path traversal)
        if "/" in client_id or ".." in client_id:
            raise HTTPException(status_code=400, detail="invalid client_id")
        data = client_svc.get_client(client_id)
        if not data:
            raise HTTPException(status_code=404, detail="client not found")
        return data

    # 3. DETECTION (minimal telemetry, no raw files)
    @app.post("/api/v1/detections")
    def submit_detection(payload: Dict[str, Any], x_client_id: Optional[str] = Header(None), authorization: Optional[str] = Header(None), request: Request = None):
        _rate_limit(request)
        ctx = _auth(x_client_id, authorization)
        # Server must not receive raw files
        if any(k in payload for k in ("raw_file", "file_bytes", "pe_bytes", "raw_bytes")):
            raise HTTPException(status_code=400, detail="raw files must not be uploaded")
        try:
            return detection_svc.submit_telemetry(payload)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/v1/detections")
    def list_detections(limit: int = 50, x_client_id: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
        ctx = _auth(x_client_id, authorization)
        return detection_svc.list_telemetry(limit=min(limit, 100))

    # 4. FEDERATED LEARNING
    @app.get("/api/v1/fl/algorithms")
    def fl_algorithms():
        return {"algorithms": fl_svc.algorithms()}

    @app.get("/api/v1/fl/configs")
    def fl_configs():
        return {"configs": fl_svc.list_configs()}

    @app.post("/api/v1/fl/experiments")
    def start_fl_experiment(payload: Dict[str, Any], x_client_id: Optional[str] = Header(None), authorization: Optional[str] = Header(None), request: Request = None):
        _rate_limit(request)
        ctx = _auth(x_client_id, authorization)
        if not _security.authorize(ctx, "experiment_management"):
            raise HTTPException(status_code=403, detail="admin required")
        return fl_svc.start_experiment(payload)

    @app.get("/api/v1/fl/experiments/{exp_id}/status")
    def fl_status(exp_id: str, request: Request = None):
        # Safe FS: exp_id must not contain path traversal
        if "/" in exp_id or ".." in exp_id:
            raise HTTPException(status_code=400, detail="invalid experiment id")
        data = fl_svc.get_status(exp_id)
        if not data:
            raise HTTPException(status_code=404, detail="experiment not found")
        return data

    @app.get("/api/v1/fl/experiments/{exp_id}/metrics/rounds")
    def fl_round_metrics(exp_id: str):
        if "/" in exp_id or ".." in exp_id:
            raise HTTPException(status_code=400, detail="invalid experiment id")
        data = fl_svc.get_round_metrics(exp_id)
        if data is None:
            raise HTTPException(status_code=404, detail="not found")
        return data

    @app.get("/api/v1/fl/experiments/{exp_id}/metrics/clients")
    def fl_client_metrics(exp_id: str):
        if "/" in exp_id or ".." in exp_id:
            raise HTTPException(status_code=400, detail="invalid id")
        data = fl_svc.get_client_metrics(exp_id)
        if data is None:
            raise HTTPException(status_code=404, detail="not found")
        return data

    @app.get("/api/v1/fl/comparison")
    def fl_comparison():
        return fl_svc.comparison()

    # 5. RESOURCE
    @app.get("/api/v1/resource/status")
    def resource_status():
        return resource_svc.get_status()

    @app.get("/api/v1/resource/policy")
    def resource_policy():
        return resource_svc.get_policy()

    @app.get("/api/v1/resource/metrics")
    def resource_metrics():
        return resource_svc.get_metrics()

    # 6. DRIFT
    @app.get("/api/v1/drift/status")
    def drift_status():
        return drift_svc.get_status()

    @app.get("/api/v1/drift/score")
    def drift_score():
        s = drift_svc.get_status()
        return {"drift_score": s.get("drift_score")}

    @app.get("/api/v1/drift/last_event")
    def drift_last():
        ev = drift_svc.get_last_event()
        if not ev:
            raise HTTPException(status_code=404, detail="no drift event")
        return ev

    @app.get("/api/v1/drift/retraining")
    def drift_retraining():
        return drift_svc.get_retraining_events()

    @app.get("/api/v1/drift/model_version")
    def drift_model_version():
        ver = drift_svc.get_model_version()
        if not ver:
            raise HTTPException(status_code=404, detail="no model version")
        return {"model_version": ver}

    # 7. MODEL
    @app.get("/api/v1/models")
    def list_models():
        return model_svc.list_versions()

    @app.get("/api/v1/models/candidates")
    def list_candidates():
        return model_svc.list_candidates()

    @app.get("/api/v1/models/{version}/validation")
    def model_validation(version: str):
        if "/" in version or ".." in version:
            raise HTTPException(status_code=400, detail="invalid version")
        data = model_svc.get_validation(version)
        if not data:
            raise HTTPException(status_code=404, detail="model not found")
        return data

    @app.post("/api/v1/models/{version}/rollback")
    def rollback_model(version: str, x_client_id: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
        ctx = _auth(x_client_id, authorization)
        if not _security.authorize(ctx, "rollback_model"):
            raise HTTPException(status_code=403, detail="admin required")
        if "/" in version or ".." in version:
            raise HTTPException(status_code=400, detail="invalid version")
        result = model_svc.rollback(version)
        if result and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result or {"status": "rolled back"}

    return app
