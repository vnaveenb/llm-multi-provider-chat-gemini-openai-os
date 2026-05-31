"""Smoke tests — no LLM calls, no network required, safe to run in CI."""

import pytest
from fastapi.testclient import TestClient

from src.chat_core.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.unit
def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200


@pytest.mark.unit
def test_health_response_schema(client: TestClient) -> None:
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "project" in data
    assert "provider" in data
