"""
Tests for the FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_history():
    """Clear transaction history before each test."""
    client.delete("/history")
    yield
    client.delete("/history")


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAnalyzeEndpoint:
    def _payload(self, amount: float = 100.0, merchant: str = "Amazon") -> dict:
        return {
            "transaction_id": "test_tx_001",
            "user_id": "user_1",
            "timestamp": "2024-01-15T10:30:00",
            "amount": amount,
            "merchant": merchant,
            "llm_enabled": False,
        }

    def test_analyze_returns_200(self):
        response = client.post("/analyze", json=self._payload())
        assert response.status_code == 200

    def test_analyze_response_has_required_fields(self):
        response = client.post("/analyze", json=self._payload())
        data = response.json()
        assert "transaction_id" in data
        assert "decision" in data
        assert "score" in data
        assert "signals" in data

    def test_decision_is_valid(self):
        response = client.post("/analyze", json=self._payload())
        data = response.json()
        assert data["decision"] in ("ignore", "monitor", "alert")

    def test_score_is_float_between_0_and_1(self):
        response = client.post("/analyze", json=self._payload())
        score = response.json()["score"]
        assert 0.0 <= score <= 1.0

    def test_large_amount_raises_risk(self):
        normal = client.post("/analyze", json=self._payload(amount=50.0))
        suspicious = client.post("/analyze", json=self._payload(amount=50000.0))
        assert suspicious.json()["score"] >= normal.json()["score"]

    def test_llm_disabled_no_llm_assessment(self):
        response = client.post("/analyze", json=self._payload())
        data = response.json()
        assert data.get("llm_assessment") is None

    def test_invalid_amount_returns_422(self):
        payload = self._payload()
        payload["amount"] = -100.0
        response = client.post("/analyze", json=payload)
        assert response.status_code == 422

    def test_missing_field_returns_422(self):
        payload = {"user_id": "user_1", "amount": 100.0}
        response = client.post("/analyze", json=payload)
        assert response.status_code == 422


class TestClearHistoryEndpoint:
    def test_clear_history_returns_200(self):
        response = client.delete("/history")
        assert response.status_code == 200
        assert response.json() == {"status": "cleared"}
