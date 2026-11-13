#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CLUSTER_NAME="${KIND_CLUSTER_NAME:-mlops-lab}"
KSERVE_VERSION="${KSERVE_VERSION:-v0.21.0}"
CERT_MANAGER_VERSION="${CERT_MANAGER_VERSION:-v1.17.0}"

echo "=========================================================="
echo "    Installing KServe (${KSERVE_VERSION}, Standard Mode) on Kind (${CLUSTER_NAME})"
echo "=========================================================="

for cmd in kubectl kind; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: '${cmd}' is not installed or not in PATH."
    exit 1
  fi
done

if ! kind get clusters | grep -qx "${CLUSTER_NAME}"; then
  echo "ERROR: Kind cluster '${CLUSTER_NAME}' is not running."
  echo "Run 'make k8s-up' first."
  exit 1
fi

kubectl config use-context "kind-${CLUSTER_NAME}" >/dev/null

echo "[1/5] Installing cert-manager (${CERT_MANAGER_VERSION})..."
kubectl apply -f "https://github.com/cert-manager/cert-manager/releases/download/${CERT_MANAGER_VERSION}/cert-manager.yaml"
kubectl rollout status deployment/cert-manager -n cert-manager --timeout=180s
kubectl rollout status deployment/cert-manager-cainjector -n cert-manager --timeout=180s
kubectl rollout status deployment/cert-manager-webhook -n cert-manager --timeout=180s
echo "Waiting for cert-manager-webhook TLS CA bundle..."
for i in $(seq 1 30); do
  if [ -n "$(kubectl get validatingwebhookconfiguration cert-manager-webhook -o jsonpath='{.webhooks[0].clientConfig.caBundle}' 2>/dev/null || true)" ]; then
    break
  fi
  sleep 2
done

echo "[2/5] Installing KServe CRDs, controller, and cluster runtimes (${KSERVE_VERSION})..."
kubectl create namespace kserve --dry-run=client -o yaml | kubectl apply -f -
kubectl apply --server-side --force-conflicts -f "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve-crds.yaml"
for attempt in 1 2 3; do
  if kubectl apply --server-side --force-conflicts -f "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve.yaml"; then
    break
  fi
  echo "Retrying kserve.yaml apply (attempt ${attempt}/3)..."
  sleep 5
done
kubectl rollout status deployment/kserve-controller-manager -n kserve --timeout=240s
echo "Waiting for KServe webhook CA bundles..."
for i in $(seq 1 30); do
  if [ -n "$(kubectl get validatingwebhookconfiguration clusterservingruntime.serving.kserve.io -o jsonpath='{.webhooks[0].clientConfig.caBundle}' 2>/dev/null || true)" ]; then
    break
  fi
  sleep 2
done
kubectl rollout status deployment/llmisvc-controller-manager -n kserve --timeout=180s || true
kubectl apply --server-side --force-conflicts -f "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve-cluster-resources.yaml" || true
kubectl get clusterservingruntime kserve-sklearnserver >/dev/null
# Keep Kind lightweight by scaling down optional LLM/localmodel controllers after applying cluster resources
kubectl scale deployment/llmisvc-controller-manager deployment/kserve-localmodel-controller-manager -n kserve --replicas=0 2>/dev/null || true

echo "[3/5] Configuring KServe Standard (RawDeployment) mode..."
kubectl patch configmap/inferenceservice-config -n kserve --type=merge \
  -p '{"data":{"deploy":"{\"defaultDeploymentMode\":\"Standard\"}"}}'

echo "[4/5] Uploading model artifact for KServe to Kind MinIO (http://localhost:30900)..."
if [ -x "${ROOT_DIR}/.venv/bin/python" ] && [ -f "${ROOT_DIR}/models/full_pipeline.joblib" ]; then
  MINIO_S3_ENDPOINT="${MINIO_S3_ENDPOINT:-http://localhost:30900}" \
  AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minioadmin}" \
  AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-minioadmin}" \
  AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
  "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/push_model_minio.py"
else
  echo "WARNING: .venv or models/full_pipeline.joblib missing; run 'make train' then 'MINIO_S3_ENDPOINT=http://localhost:30900 python scripts/push_model_minio.py'."
fi

echo "[5/5] Deploying InferenceService 'house-price-kserve' and Kubeflow Models Web App UI..."
kubectl apply -f "${ROOT_DIR}/k8s/kserve/inference-service.yaml"
kubectl wait --for=condition=Ready inferenceservice/house-price-kserve -n mlops-lab --timeout=240s
kubectl apply -k "https://github.com/kserve/models-web-app/manifests/kustomize/base?ref=v1.1.1"
kubectl rollout status deployment/kserve-models-web-application -n kserve --timeout=180s

echo ""
echo "=========================================================="
echo "    KServe InferenceService 'house-price-kserve' is READY!"
echo "=========================================================="
kubectl get inferenceservice,pods -n mlops-lab

