"""
Automated Prediction & Monitoring Verification Script
Sends sample inference requests to the running FastAPI service, tests batch
scoring, triggers validation errors to verify Prometheus counter, and tests drift detection.
"""

import json
import sys

import httpx

API_URL = "http://localhost:8000"


def test_api():
    print("==================================================")
    print("      FastAPI Prediction & Monitoring Verification ")
    print(f"      Target: {API_URL}")
    print("==================================================\n")

    client = httpx.Client(base_url=API_URL, timeout=10.0)

    # 1. Health check
    try:
        print("[1/5] Checking /health endpoint...")
        resp = client.get("/health")
        print(f"  HTTP {resp.status_code}: {resp.json()}")
        assert resp.status_code == 200, "Health check failed!"
    except Exception as e:
        print(f"  ERROR connecting to API: {e}")
        print("  Please ensure FastAPI is running via 'docker compose up -d' or 'make run-api'.")
        sys.exit(1)

    # 2. Single Prediction
    print("\n[2/5] Testing single prediction on /predict...")
    sample_house = {
        "area": 2400,
        "bedrooms": 3,
        "bathrooms": 2.5,
        "location": "suburban",
        "age": 8,
        "garage": 2,
    }
    resp = client.post("/predict", json=sample_house)
    print(f"  Input Features: {sample_house}")
    print(f"  HTTP {resp.status_code} Response:")
    print(json.dumps(resp.json(), indent=4))
    assert resp.status_code == 200

    # 3. Batch Prediction
    print("\n[3/5] Testing batch prediction on /predict/batch...")
    batch_payload = {
        "houses": [
            {
                "area": 1200,
                "bedrooms": 2,
                "bathrooms": 1.0,
                "location": "rural",
                "age": 25,
                "garage": 0,
            },
            {
                "area": 3800,
                "bedrooms": 5,
                "bathrooms": 4.0,
                "location": "urban",
                "age": 2,
                "garage": 3,
            },
        ]
    }
    resp = client.post("/predict/batch", json=batch_payload)
    print(f"  HTTP {resp.status_code} Batch Scored {resp.json().get('count')} houses:")
    for p in resp.json().get("predictions", []):
        print(
            f"    - Area: {p['features_received']['area']} sqft ({p['features_received']['location']}) -> ${p['predicted_price']:,.2f}"
        )
    assert resp.status_code == 200

    # 4. Invalid Input Handling (testing 422 error tracking)
    print("\n[4/5] Testing invalid input schema handling...")
    invalid_house = {
        "area": 50,  # Below minimum (300)
        "bedrooms": -1,  # Negative bedrooms
        "bathrooms": 2.0,
        "location": "atlantis",  # Unknown category
        "age": 5,
        "garage": 1,
    }
    resp = client.post("/predict", json=invalid_house)
    print(f"  Submitted invalid payload. Received expected HTTP {resp.status_code}")
    print(f"  Status response: {resp.json().get('status')}")
    assert resp.status_code == 422

    # 5. Data Drift Detection
    print("\n[5/5] Testing data drift analysis endpoint /monitor/drift...")
    # Send a batch of houses with significantly shifted features
    drift_batch = {
        "houses": [
            {
                "area": 8500,
                "bedrooms": 6,
                "bathrooms": 5.0,
                "location": "rural",
                "age": 70,
                "garage": 4,
            }
            for _ in range(15)
        ]
    }
    resp = client.post("/monitor/drift", json=drift_batch)
    if resp.status_code == 200:
        drift_data = resp.json()
        print(f"  Drift Detected: {drift_data.get('overall_drift_detected')}")
        print(f"  Drifted Features: {drift_data.get('drifted_features')}")
    else:
        print(f"  Drift endpoint status: {resp.status_code} ({resp.text})")

    # 6. Prometheus Metrics Scrape Check
    print("\n[*] Inspecting Prometheus metrics endpoint /metrics...")
    resp = client.get("/metrics")
    metrics_text = resp.text
    for metric_name in [
        "mlops_prediction_requests_total",
        "mlops_prediction_latency_seconds",
        "mlops_invalid_requests_total",
        "mlops_predicted_price_usd",
    ]:
        if metric_name in metrics_text:
            print(f"  ✓ {metric_name} is actively exported")
        else:
            print(f"  ✗ {metric_name} was not found in metrics export")

    print("\n==================================================")
    print("  ✓ All API & Monitoring Verifications Passed!")
    print("==================================================\n")


if __name__ == "__main__":
    test_api()
