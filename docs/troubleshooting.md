# Comprehensive Troubleshooting & Operations Guide

This guide compiles common issues encountered across the MLOps lifecycle, explaining root causes and exact resolution commands.

---

## 1. Port Conflicts & macOS AirPlay Receiver

### Symptom: `Bind for 0.0.0.0:5000 failed: port is already allocated`
- **Root Cause:** macOS Monterey, Ventura, Sonoma, and Sequoia enable the **AirPlay Receiver** service by default, listening on port `5000` (`ControlCenter`).
- **Diagnosis:** Run `lsof -i :5000` in your terminal. You will see `ControlCe` occupying the port.
- **Resolution in this Lab:** We map MLflow to port `5001` in `.env`:
  ```bash
  MLFLOW_PORT=5001
  ```
  Access the MLflow UI at `http://localhost:5001` instead of `5000`.
- **Alternative System Fix:** Disable AirPlay Receiver via **macOS System Settings** -> **General** -> **AirDrop & AirPlay** -> Toggle off **AirPlay Receiver**.

---

## 2. Docker Credential Helper Keychain Error

### Symptom: `error getting credentials - err: exit status 1, out: ""`
- **Root Cause:** Docker Desktop writes `"credsStore": "desktop"` into `~/.docker/config.json`. When Docker CLI runs in non-interactive terminals or build scripts, the macOS Keychain credential helper cannot authenticate without GUI prompts.
- **Resolution:**
  Edit `~/.docker/config.json` and remove or empty the `credsStore` key:
  ```json
  {
    "auths": {},
    "currentContext": "desktop-linux"
  }
  ```

---

## 3. MinIO & S3 Storage Issues

### Symptom 1: `NoSuchBucket: The specified bucket does not exist`
- **Root Cause:** Application attempted to upload before `dvc-store` or `mlflow-artifacts` buckets were created.
- **Diagnosis:** Check `docker compose ps` to ensure `minio-init` completed with code 0.
- **Resolution:** Manually trigger the bucket initialization:
  ```bash
  docker compose run --rm minio-init
  ```

### Symptom 2: `Could not connect to the endpoint URL: http://minio:9000`
- **Root Cause:** Host vs. Container networking confusion.
  - From your local terminal (host), MinIO is reachable at `http://localhost:9000`.
  - From inside a Docker container on the bridge network, MinIO is reachable at `http://minio:9000`.
- **Resolution:** Verify `MLFLOW_S3_ENDPOINT_URL=http://minio:9000` in container configurations.

---

## 4. MLflow Server & Database Issues

### Symptom: `psycopg2.OperationalError: could not connect to server: Connection refused`
- **Root Cause:** MLflow server started before PostgreSQL was ready to accept socket connections.
- **Resolution:** Our `docker-compose.yml` uses strict Docker Compose healthchecks:
  ```yaml
  depends_on:
    postgres:
      condition: service_healthy
  ```
  PostgreSQL runs `pg_isready` before MLflow is permitted to boot.

---

## 5. DVC Data Versioning Issues

### Symptom: `ERROR: unexpected error - unable to open database file`
- **Root Cause:** Sandboxed environment restriction or SQLite lock contention on `.dvc/tmp/rw/state`.
- **Resolution:** Remove the stale temporary lock file:
  ```bash
  rm -f .dvc/tmp/rw/state*
  ```

### Symptom: `dvc push fails with 403 Forbidden`
- **Root Cause:** MinIO credentials missing in local environment.
- **Resolution:** Export the AWS credentials in your active shell:
  ```bash
  export AWS_ACCESS_KEY_ID=minioadmin
  export AWS_SECRET_ACCESS_KEY=minioadmin
  dvc push
  ```

---

## 6. FastAPI Inference Status Codes

| HTTP Status | Meaning | Typical Cause | Resolution |
| :--- | :--- | :--- | :--- |
| **`200 OK`** | Success | Request was valid and model predicted price. | Everything working normally. |
| **`422 Unprocessable`** | Schema Validation Failure | Input failed Pydantic rules (e.g. `area < 300`, invalid location). | Inspect response `details` and fix request fields. |
| **`503 Unavailable`** | Model Not Loaded | Container started without a model artifact. | Ensure `models/full_pipeline.joblib` exists or MLflow is reachable. |
| **`500 Internal Error`** | Processing Exception | Feature calculation error or unhandled bug. | Check container logs via `docker compose logs api`. |

---

## 7. Prometheus & Grafana Observability

### Symptom 1: Prometheus target `api:8000` shows `DOWN`
- **Diagnosis:** Open `http://localhost:9090/targets` in your browser.
- **Root Cause:** FastAPI is not yet running or healthy.
- **Resolution:** Verify FastAPI responds to `curl http://localhost:8000/metrics`.

### Symptom 2: Grafana Dashboard shows "No Data"
- **Diagnosis:** No inference traffic has been sent to the API yet.
- **Resolution:** Send test requests to populate Prometheus metrics:
  ```bash
  python scripts/test_prediction.py
  ```
  Within 5 seconds, Grafana will display live timeseries graphs.

