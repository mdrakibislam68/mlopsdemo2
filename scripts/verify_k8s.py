"""
End-to-end verification script for the Kind Kubernetes MLOps deployment.
Checks all NodePort endpoints, runs live FastAPI inference, and verifies Prometheus target scraping.
"""

import json
import sys

import httpx

K8S_SERVICES = [
    {
        "name": "MinIO S3 API (K8s)",
        "url": "http://localhost:30900/minio/health/live",
        "expected": [200],
    },
    {"name": "MinIO Console (K8s)", "url": "http://localhost:30901", "expected": [200, 307]},
    {"name": "MLflow Server (K8s)", "url": "http://localhost:30500", "expected": [200]},
    {"name": "FastAPI Health (K8s)", "url": "http://localhost:30800/health", "expected": [200]},
    {"name": "FastAPI Metrics (K8s)", "url": "http://localhost:30800/metrics", "expected": [200]},
    {"name": "Prometheus (K8s)", "url": "http://localhost:30909/-/healthy", "expected": [200]},
    {"name": "Grafana (K8s)", "url": "http://localhost:30300/api/health", "expected": [200]},
]


def main() -> int:
    print("==========================================================")
    print("    Verifying Kind Kubernetes MLOps Services              ")
    print("==========================================================\n")

    client = httpx.Client(timeout=5.0)
    all_ok = True

    for svc in K8S_SERVICES:
        try:
            resp = client.get(svc["url"])
            if resp.status_code in svc["expected"]:
                print(f"  ✓ {svc['name']:<26} UP & HEALTHY (HTTP {resp.status_code})")
            else:
                print(f"  ✗ {svc['name']:<26} UNEXPECTED STATUS (HTTP {resp.status_code})")
                all_ok = False
        except Exception as exc:
            print(f"  ✗ {svc['name']:<26} UNREACHABLE ({exc})")
            all_ok = False

    if not all_ok:
        return 1

    print("\n[*] Checking /health payload & running /predict on FastAPI (port 30800)...")
    health = client.get("http://localhost:30800/health").json()
    print(f"  Health: {json.dumps(health)}")

    payload = {
        "area": 2400,
        "bedrooms": 3,
        "bathrooms": 2.5,
        "location": "suburban",
        "age": 8,
        "garage": 2,
    }
    pred_resp = client.post("http://localhost:30800/predict", json=payload)
    pred_resp.raise_for_status()
    print(f"  Prediction: {json.dumps(pred_resp.json())}")

    print("\n[*] Checking Prometheus scrape targets (port 30909)...")
    targets = client.get("http://localhost:30909/api/v1/targets").json()
    active = [(t["labels"]["job"], t["health"]) for t in targets["data"]["activeTargets"]]
    print(f"  Active Targets: {active}")

    if ("fastapi", "up") not in active:
        print("  ! Warning: 'fastapi' target not yet marked 'up' by Prometheus.")
        return 1

    print("\n  ✓ All Kind Kubernetes checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
