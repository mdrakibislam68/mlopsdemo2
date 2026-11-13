#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CLUSTER_NAME="${KIND_CLUSTER_NAME:-mlops-lab}"
KFP_VERSION="${KFP_VERSION:-2.18.0}"

echo "=========================================================="
echo "    Installing Standalone Kubeflow Pipelines (${KFP_VERSION})"
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

echo "[1/4] Creating 'kubeflow' namespace..."
kubectl create namespace kubeflow --dry-run=client -o yaml | kubectl apply -f -

echo "[2/4] Applying Kubeflow Pipelines platform-agnostic manifests (v${KFP_VERSION})..."
# Ensure Argo Workflows CRDs are in place
kubectl apply -k "https://github.com/argoproj/argo-workflows/manifests/cluster-install?ref=v3.5.14" 2>&1 | grep "customresourcedefinition" || true

# Apply CRDs and resources (using server-side apply with retry for large manifests)
for attempt in 1 2 3; do
  if kubectl apply --server-side --force-conflicts -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${KFP_VERSION}"; then
    break
  fi
  echo "Retrying KFP apply (attempt ${attempt}/3)..."
  sleep 10
done

# Configure workflow controller metrics port to avoid collision and scale down optional x86-only cache/metadata-writer on ARM Mac
kubectl patch configmap workflow-controller-configmap -n kubeflow --type=merge -p '{"data":{"metricsConfig":"{\"port\":9091}"}}' 2>/dev/null || true
kubectl scale deployment/metadata-writer deployment/cache-server -n kubeflow --replicas=0 2>/dev/null || true

echo "[3/4] Waiting for core KFP services to become Ready..."
kubectl rollout status deployment/ml-pipeline-ui -n kubeflow --timeout=240s
kubectl rollout status deployment/ml-pipeline -n kubeflow --timeout=240s
kubectl rollout status deployment/workflow-controller -n kubeflow --timeout=240s

echo "[4/4] Kubeflow Pipelines deployment status:"
kubectl get pods,svc -n kubeflow

echo ""
echo "=========================================================="
echo "    Kubeflow Pipelines is READY!                          "
echo "=========================================================="
echo "Open UI: make kfp-ui (Access at http://localhost:8080)"

