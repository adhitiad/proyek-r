from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings


def test_root_smoke():
    # Ensure API key doesn't block basic smoke tests
    settings.API_KEY = ""
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json().get("message")


def test_metrics_smoke():
    settings.API_KEY = ""
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
