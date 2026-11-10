#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "    Executing Full End-to-End MLOps Pipeline              "
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${ROOT_DIR}/.venv/bin/python"
if [ ! -f "${PYTHON}" ]; then
  PYTHON="python3"
fi

echo "[Step 1/4] Generating reproducible synthetic dataset..."
"${PYTHON}" src/data/generate_data.py --config params.yaml

echo -e "\n[Step 2/4] Validating dataset schema and ranges..."
"${PYTHON}" src/data/validate_data.py --config params.yaml

echo -e "\n[Step 3/4] Preprocessing features and splitting train/test..."
"${PYTHON}" src/data/preprocess.py --config params.yaml

echo -e "\n[Step 4/4] Training models, comparing performance, and logging to MLflow..."
"${PYTHON}" src/training/train.py --config params.yaml

echo -e "\n=========================================================="
echo "  ✓ Pipeline completed successfully!                      "
echo "  Artifacts saved in data/processed/ and models/          "
echo "  Metrics saved in evaluation_metrics.json                "
echo "=========================================================="

