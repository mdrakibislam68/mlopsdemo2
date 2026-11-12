#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CLUSTER_NAME="${KIND_CLUSTER_NAME:-mlops-lab}"
NAMESPACE="mlops-lab"

echo "=========================================================="
echo "    Deploying End-to-End MLOps Lab to Kind Kubernetes     "
echo "=========================================================="

# 1. Ensure required CLIs exist
for cmd in docker kind kubectl; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: '${cmd}' is not installed or not in PATH."
    exit 1
  fi
done

# 2. Load local .env (or fallback to .env.example) for K8s Secret generation
ENV_FILE="${ROOT_DIR}/.env"
if [ ! -f "${ENV_FILE}" ]; then
  ENV_FILE="${ROOT_DIR}/.env.example"
fi
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

# 3. Create Kind cluster if not already running
if kind get clusters | grep -qx "${CLUSTER_NAME}"; then
  echo "[1/6] Kind cluster '${CLUSTER_NAME}' already exists."
else
  echo "[1/6] Creating multi-node Kind cluster '${CLUSTER_NAME}'..."
  kind create cluster --config k8s/kind-config.yaml
fi

kubectl config use-context "kind-${CLUSTER_NAME}" >/dev/null

# 4. Ensure custom images exist locally and load all images into Kind nodes
echo "[2/6] Checking and building custom Docker images if needed..."
if ! docker image inspect mlops-fastapi-service:1.0.0 >/dev/null 2>&1 || \
   ! docker image inspect mlops-mlflow-server:1.0.0 >/dev/null 2>&1; then
  docker compose build api mlflow
fi

# Only custom local images must be loaded into Kind; public images pull cleanly from registries
# (avoids Docker Desktop multi-arch manifest digest issues during `kind load` on public images).
IMAGES=(
  "mlops-fastapi-service:1.0.0"
  "mlops-mlflow-server:1.0.0"
)

echo "[3/6] Loading custom Docker images into Kind cluster '${CLUSTER_NAME}'..."
for img in "${IMAGES[@]}"; do
  echo "  Loading ${img}..."
  kind load docker-image "${img}" --name "${CLUSTER_NAME}"
done

# 5. Apply Namespace, Secrets, and ConfigMaps from configs/
echo "[4/6] Applying Namespace, Secrets, and ConfigMaps..."
kubectl apply -f k8s/namespace.yaml

kubectl create secret generic mlops-secrets \
  --namespace="${NAMESPACE}" \
  --from-literal=POSTGRES_USER="${POSTGRES_USER:-mlflow}" \
  --from-literal=POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-mlflowpassword}" \
  --from-literal=POSTGRES_DB="${POSTGRES_DB:-mlflow}" \
  --from-literal=MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}" \
  --from-literal=MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin}" \
  --from-literal=AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minioadmin}" \
  --from-literal=AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-minioadmin}" \
  --from-literal=GF_SECURITY_ADMIN_USER="${GRAFANA_ADMIN_USER:-admin}" \
  --from-literal=GF_SECURITY_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-admin}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap prometheus-config \
  --namespace="${NAMESPACE}" \
  --from-file=prometheus.yml=configs/prometheus.yml \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap grafana-datasource \
  --namespace="${NAMESPACE}" \
  --from-file=prometheus.yml=configs/grafana/provisioning/datasources/prometheus.yml \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap grafana-dashboard-provider \
  --namespace="${NAMESPACE}" \
  --from-file=dashboards.yml=configs/grafana/provisioning/dashboards/dashboards.yml \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap grafana-dashboards \
  --namespace="${NAMESPACE}" \
  --from-file=mlops_dashboard.json=configs/grafana/dashboards/mlops_dashboard.json \
  --dry-run=client -o yaml | kubectl apply -f -

# 6. Apply Kustomize manifests (recreate Job safely on idempotent runs)
echo "[5/6] Applying Kubernetes manifests via Kustomize..."
kubectl delete job minio-init -n "${NAMESPACE}" --ignore-not-found=true >/dev/null
kubectl apply -k k8s/

# 7. Wait for readiness
echo "[6/6] Waiting for workloads to become ready..."
kubectl rollout status deployment/postgres -n "${NAMESPACE}" --timeout=120s
kubectl rollout status deployment/minio -n "${NAMESPACE}" --timeout=120s
kubectl wait --for=condition=complete job/minio-init -n "${NAMESPACE}" --timeout=120s
kubectl rollout status deployment/mlflow -n "${NAMESPACE}" --timeout=180s
kubectl rollout status deployment/api -n "${NAMESPACE}" --timeout=120s
kubectl rollout status deployment/prometheus -n "${NAMESPACE}" --timeout=120s
kubectl rollout status deployment/grafana -n "${NAMESPACE}" --timeout=120s

echo ""
echo "=========================================================="
echo "  Kind Kubernetes Cluster '${CLUSTER_NAME}' is READY!     "
echo "=========================================================="
echo "  FastAPI Docs : http://localhost:30800/docs"
echo "  MLflow UI    : http://localhost:30500"
echo "  MinIO Console: http://localhost:30901"
echo "  Prometheus   : http://localhost:30909"
echo "  Grafana      : http://localhost:30300"
echo "=========================================================="

