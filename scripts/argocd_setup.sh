#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CLUSTER_NAME="${KIND_CLUSTER_NAME:-mlops-lab}"
ARGOCD_NAMESPACE="argocd"
ARGOCD_MANIFEST_URL="https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml"

echo "=========================================================="
echo "    Installing & Configuring Argo CD on Kind (${CLUSTER_NAME})"
echo "=========================================================="

for cmd in kubectl kind; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: '${cmd}' is not installed or not in PATH."
    exit 1
  fi
done

if ! kind get clusters | grep -qx "${CLUSTER_NAME}"; then
  echo "ERROR: Kind cluster '${CLUSTER_NAME}' is not running."
  echo "Run 'make k8s-up' first to create the cluster, deploy secrets/configmaps, and load images."
  exit 1
fi

kubectl config use-context "kind-${CLUSTER_NAME}" >/dev/null

echo "[1/4] Creating namespace '${ARGOCD_NAMESPACE}' (if needed)..."
kubectl create namespace "${ARGOCD_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

echo "[2/4] Installing Argo CD upstream manifests..."
kubectl apply -n "${ARGOCD_NAMESPACE}" --server-side --force-conflicts -f "${ARGOCD_MANIFEST_URL}"

echo "[3/4] Waiting for Argo CD core deployments to become Ready..."
kubectl rollout status deployment/argocd-server -n "${ARGOCD_NAMESPACE}" --timeout=240s
kubectl rollout status deployment/argocd-repo-server -n "${ARGOCD_NAMESPACE}" --timeout=240s
kubectl rollout status statefulset/argocd-application-controller -n "${ARGOCD_NAMESPACE}" --timeout=240s

echo "[4/4] Applying Argo CD Application manifest (k8s/argocd/application.yaml)..."
kubectl apply -f "${ROOT_DIR}/k8s/argocd/application.yaml"

echo ""
echo "=========================================================="
echo "    Argo CD is installed and tracking branch 'v2' (k8s/)  "
echo "=========================================================="
echo "UI Port-Forward : make argocd-ui   (opens https://localhost:8081)"
echo "Username        : admin"
echo -n "Initial Password: "
kubectl -n "${ARGOCD_NAMESPACE}" get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
echo ""

