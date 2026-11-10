#!/bin/sh
set -e

echo "[MinIO-Init] Waiting for MinIO S3 API to become available..."
until /usr/bin/mc alias set localminio http://minio:9000 "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}"; do
  echo "[MinIO-Init] MinIO is starting up; retrying in 2 seconds..."
  sleep 2
done

echo "[MinIO-Init] Connected successfully to MinIO."

# Create DVC storage bucket
if /usr/bin/mc ls localminio/"${MINIO_DVC_BUCKET:-dvc-store}" >/dev/null 2>&1; then
  echo "[MinIO-Init] Bucket '${MINIO_DVC_BUCKET:-dvc-store}' already exists."
else
  echo "[MinIO-Init] Creating bucket '${MINIO_DVC_BUCKET:-dvc-store}'..."
  /usr/bin/mc mb localminio/"${MINIO_DVC_BUCKET:-dvc-store}"
fi

# Create MLflow artifacts bucket
if /usr/bin/mc ls localminio/"${MINIO_MLFLOW_BUCKET:-mlflow-artifacts}" >/dev/null 2>&1; then
  echo "[MinIO-Init] Bucket '${MINIO_MLFLOW_BUCKET:-mlflow-artifacts}' already exists."
else
  echo "[MinIO-Init] Creating bucket '${MINIO_MLFLOW_BUCKET:-mlflow-artifacts}'..."
  /usr/bin/mc mb localminio/"${MINIO_MLFLOW_BUCKET:-mlflow-artifacts}"
fi

echo "[MinIO-Init] Buckets initialized successfully."
exit 0

