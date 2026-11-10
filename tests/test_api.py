"""
Integration tests for FastAPI endpoints: health, prediction, metrics, error handling.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Verify landing endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "documentation" in data
    assert "metrics" in data


def test_health_endpoint(client):
    """Verify health check endpoint returns 200 and model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "uptime_seconds" in data
    assert "model_name" in data


def test_predict_endpoint_valid(client):
    """Verify single prediction endpoint with valid payload."""
    payload = {
        "area": 2100,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "location": "suburban",
        "age": 10,
        "garage": 2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert data["predicted_price"] > 0
    assert data["currency"] == "USD"
    assert data["model_name"] == "house-price-predictor"


def test_predict_endpoint_invalid_location(client):
    """Verify 422 error when unknown location is submitted."""
    payload = {
        "area": 2000,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "location": "outer_space",  # Invalid
        "age": 5,
        "garage": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "validation_failed"


def test_predict_endpoint_out_of_range(client):
    """Verify 422 error when area is below minimum."""
    payload = {
        "area": 50,  # Below minimum (300)
        "bedrooms": 3,
        "bathrooms": 2.0,
        "location": "urban",
        "age": 5,
        "garage": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch_endpoint(client):
    """Verify batch prediction endpoint scores all items."""
    payload = {
        "houses": [
            {
                "area": 1800,
                "bedrooms": 2,
                "bathrooms": 1.5,
                "location": "urban",
                "age": 15,
                "garage": 1,
            },
            {
                "area": 3200,
                "bedrooms": 4,
                "bathrooms": 3.0,
                "location": "rural",
                "age": 5,
                "garage": 2,
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    for p in data["predictions"]:
        assert p["predicted_price"] > 0


def test_prometheus_metrics_endpoint(client):
    """Verify Prometheus metrics are exposed and include expected counters."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert "mlops_prediction_requests_total" in body
    assert "mlops_prediction_latency_seconds" in body
    assert "mlops_api_uptime_seconds" in body
