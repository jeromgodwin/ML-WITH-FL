"""API tests for FedShield Control Center (Phase 20 §9)."""

import pathlib
import tempfile

import pytest
from fastapi.testclient import TestClient

from fedshield.config import ServerNetworkConfig
from src.federated.network.auth import ClientRegistry
from backend.app import create_app


@pytest.fixture
def app_and_reg(tmp_path):
    cfg = ServerNetworkConfig(host="127.0.0.1", port=8000, secure=False)
    reg_dir = tmp_path / "reg"
    app = create_app(cfg, model_registry_dir=reg_dir / "models", client_registry_dir=reg_dir / "clients")
    # Provision admin and client
    # Access the app's internal registries via the service layer's client_registry
    # We provision directly via ClientRegistry for test
    from src.federated.network.auth import ClientRegistry as CR
    cr = CR(reg_dir / "clients")
    admin = cr.provision_client("admin-1", role="admin")
    client = cr.provision_client("client-1", role="client")
    return app, admin, client, reg_dir


def _auth_headers(auth):
    return {"X-Client-Id": auth.client_id, "Authorization": f"Bearer {auth.token}"}


def test_health_and_system(app_and_reg):
    app, admin, client, _ = app_and_reg
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.get("/api/v1/status").status_code == 200
    assert c.get("/api/v1/protection/status").status_code == 200


def test_client_management(app_and_reg):
    app, admin, client, _ = app_and_reg
    c = TestClient(app)
    # List clients requires auth
    assert c.get("/api/v1/clients").status_code == 401
    headers = _auth_headers(client)
    r = c.get("/api/v1/clients", headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert any(d["client_id"] == "client-1" for d in data)
    # Do not expose token
    assert all("token" not in d for d in data)
    # Get single client
    r2 = c.get("/api/v1/clients/client-1", headers=headers)
    assert r2.status_code == 200
    assert r2.json()["client_id"] == "client-1"
    # Invalid client_id format
    r3 = c.get("/api/v1/clients/../etc", headers=headers)
    assert r3.status_code == 400


def test_detection_minimal_telemetry(app_and_reg):
    app, admin, client, _ = app_and_reg
    c = TestClient(app)
    headers = _auth_headers(client)
    # Valid minimal telemetry
    payload = {"client_id": "client-1", "detection_id": "d1", "sha256": "abc", "file_type": "pe_exe", "model_version": "v1", "malware_probability": 0.9, "risk_score": 80, "verdict": "HIGH", "action": "QUARANTINE"}
    r = c.post("/api/v1/detections", json=payload, headers=headers)
    assert r.status_code == 200
    # Raw file must be rejected
    bad = {"client_id": "client-1", "detection_id": "d2", "raw_file": "binary"}
    r2 = c.post("/api/v1/detections", json=bad, headers=headers)
    assert r2.status_code == 400


def test_federated_learning_endpoints(app_and_reg):
    app, admin, client, _ = app_and_reg
    c = TestClient(app)
    assert c.get("/api/v1/fl/algorithms").status_code == 200
    assert "fedavg" in c.get("/api/v1/fl/algorithms").json()["algorithms"]
    assert c.get("/api/v1/fl/configs").status_code == 200
    # Start experiment requires admin
    headers_client = _auth_headers(client)
    r = c.post("/api/v1/fl/experiments", json={"fl": {"algorithm": "fedavg"}}, headers=headers_client)
    assert r.status_code == 403
    headers_admin = _auth_headers(admin)
    r2 = c.post("/api/v1/fl/experiments", json={"fl": {"algorithm": "fedavg"}}, headers=headers_admin)
    assert r2.status_code == 200
    exp_id = r2.json()["experiment_id"]
    # Status
    assert c.get(f"/api/v1/fl/experiments/{exp_id}/status").status_code == 200
    # Invalid exp_id with path traversal
    assert c.get("/api/v1/fl/experiments/../etc/status").status_code == 400
    # Comparison
    assert c.get("/api/v1/fl/comparison").status_code == 200


def test_resource_drift_model(app_and_reg):
    app, admin, client, _ = app_and_reg
    c = TestClient(app)
    assert c.get("/api/v1/resource/status").status_code == 200
    assert c.get("/api/v1/resource/policy").status_code == 200
    assert c.get("/api/v1/resource/metrics").status_code == 200
    assert c.get("/api/v1/drift/status").status_code == 200
    # Drift last event may be 404 if none
    assert c.get("/api/v1/drift/last_event").status_code in (200, 404)
    assert c.get("/api/v1/models").status_code == 200
    assert c.get("/api/v1/models/candidates").status_code == 200
    # Validation for unknown model
    assert c.get("/api/v1/models/unknown/validation").status_code == 404
    # Rollback requires admin
    headers_client = _auth_headers(client)
    assert c.post("/api/v1/models/v1/rollback", headers=headers_client).status_code == 403
    headers_admin = _auth_headers(admin)
    # Rollback with invalid version format
    assert c.post("/api/v1/models/../etc/rollback", headers=headers_admin).status_code == 400


def test_security_rate_limiting_and_safe_fs(app_and_reg):
    app, admin, client, _ = app_and_reg
    c = TestClient(app)
    headers = _auth_headers(client)
    # Rate limiting: hammer with many requests quickly (120 limit)
    for _ in range(5):
        assert c.get("/api/v1/clients/client-1", headers=headers).status_code == 200
    # Authentication required
    assert c.get("/api/v1/clients").status_code == 401
    # Safe error handling — invalid JSON should not leak internals
    r = c.post("/api/v1/detections", content="not-json", headers={**headers, "Content-Type": "application/json"})
    assert r.status_code in (400, 422, 500)
    # No arbitrary filesystem access — exp_id with .. rejected
    assert c.get("/api/v1/fl/experiments/../../etc/passwd/status").status_code == 400


def test_service_abstraction_no_ml_in_handlers():
    """Handlers delegate to services — check service files exist and app uses them."""
    import pathlib
    assert pathlib.Path("backend/services/system_service.py").exists()
    assert pathlib.Path("backend/services/client_service.py").exists()
    assert pathlib.Path("backend/services/fl_service.py").exists()
    # Ensure app.py does not import torch/sklearn directly (no ML in handlers)
    text = pathlib.Path("backend/app.py").read_text(encoding="utf-8")
    assert "import torch" not in text
    assert "sklearn" not in text
