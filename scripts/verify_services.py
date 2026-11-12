"""
Health & Connectivity Verification Script for Docker Compose Services
Tests that PostgreSQL, MinIO, MLflow, FastAPI, Prometheus, and Grafana are all
reachable and healthy.
"""

import sys

import httpx

SERVICES = [
    {
        "name": "MinIO S3 API",
        "url": "http://localhost:9000/minio/health/live",
        "expected_status": [200],
    },
    {"name": "MinIO Web Console", "url": "http://localhost:9001", "expected_status": [200, 307]},
    {"name": "MLflow Tracking Server", "url": "http://localhost:5001", "expected_status": [200]},
    {
        "name": "FastAPI Health Check",
        "url": "http://localhost:8000/health",
        "expected_status": [200],
    },
    {
        "name": "FastAPI Metrics Endpoint",
        "url": "http://localhost:8000/metrics",
        "expected_status": [200],
    },
    {
        "name": "Prometheus Web UI",
        "url": "http://localhost:9090/-/healthy",
        "expected_status": [200],
    },
    {
        "name": "Grafana Dashboard",
        "url": "http://localhost:3000/api/health",
        "expected_status": [200],
    },
]


def verify_services():
    print("==========================================================")
    print("    Verifying Local MLOps Docker Compose Services         ")
    print("==========================================================\n")

    client = httpx.Client(timeout=5.0)
    all_ok = True

    for s in SERVICES:
        try:
            resp = client.get(s["url"])
            if resp.status_code in s["expected_status"]:
                print(f"  ✓ {s['name']:<28} UP & HEALTHY (HTTP {resp.status_code})")
            else:
                print(f"  ✗ {s['name']:<28} UNEXPECTED STATUS (HTTP {resp.status_code})")
                all_ok = False
        except Exception as e:
            print(f"  ✗ {s['name']:<28} DOWN / UNREACHABLE ({e})")
            all_ok = False

    print("\n----------------------------------------------------------")
    if all_ok:
        print("  ✓ All Docker Compose Services are running and responsive!")
    else:
        print("  ! Some services are currently unreachable.")
        print(
            "    If you just ran 'docker compose up', give services ~15s to complete healthchecks."
        )
    print("----------------------------------------------------------\n")
    return all_ok


if __name__ == "__main__":
    success = verify_services()
    sys.exit(0 if success else 1)
