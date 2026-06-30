"""Integration tests for the FastAPI server (no model file needed)."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("server.main.ONNX_PATH") as mock_path:
        mock_path.exists.return_value = False
        from server.main import app
        yield TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_metrics_missing_session(client):
    resp = client.get("/metrics/nonexistent-session")
    assert resp.status_code == 200
    assert "error" in resp.json()
