.PHONY: help setup data validate preprocess train test lint run-api docker-up docker-down docker-build dvc-push dvc-pull push-model predict-test clean

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PYTEST ?= .venv/bin/pytest
RUFF ?= .venv/bin/ruff
DVC ?= .venv/bin/dvc

help:
	@echo "=========================================================================="
	@echo "                   End-to-End MLOps Lab Management                        "
	@echo "=========================================================================="
	@echo "  make setup         - Create venv and install dependencies"
	@echo "  make data          - Generate reproducible synthetic house price dataset"
	@echo "  make validate      - Validate schema and constraints on raw data"
	@echo "  make preprocess    - Split train/test and build feature preprocessor"
	@echo "  make train         - Train models, log to MLflow, and register champion"
	@echo "  make pipeline      - Run full data & training pipeline end-to-end"
	@echo "  make test          - Run Pytest unit and integration test suite"
	@echo "  make lint          - Lint and format code with Ruff"
	@echo "  make run-api       - Run FastAPI service locally outside Docker"
	@echo "  make docker-build  - Build API & MLflow Docker images"
	@echo "  make docker-up     - Start all 6 MLOps services via Docker Compose"
	@echo "  make docker-down   - Stop Docker Compose services"
	@echo "  make dvc-push      - Push tracked dataset version to MinIO remote"
	@echo "  make dvc-pull      - Pull dataset version from MinIO remote"
	@echo "  make push-model    - Upload local models/ artifacts to MinIO model/ folder"
	@echo "  make predict-test  - Send sample prediction request to running API"
	@echo "  make clean         - Clean temporary cache and log files"
	@echo "=========================================================================="

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) src/data/generate_data.py --config params.yaml

validate:
	$(PYTHON) src/data/validate_data.py --config params.yaml

preprocess:
	$(PYTHON) src/data/preprocess.py --config params.yaml

train:
	MLFLOW_TRACKING_URI=$${MLFLOW_TRACKING_URI:-http://localhost:5001} \
	MLFLOW_S3_ENDPOINT_URL=$${MLFLOW_S3_ENDPOINT_URL:-http://localhost:9000} \
	AWS_ACCESS_KEY_ID=$${AWS_ACCESS_KEY_ID:-minioadmin} \
	AWS_SECRET_ACCESS_KEY=$${AWS_SECRET_ACCESS_KEY:-minioadmin} \
	AWS_DEFAULT_REGION=$${AWS_DEFAULT_REGION:-us-east-1} \
	$(PYTHON) src/training/train.py --config params.yaml

pipeline: data validate preprocess train

test:
	$(PYTEST) -v

lint:
	$(RUFF) check src/ api/ tests/
	$(RUFF) format --check src/ api/ tests/

format:
	$(RUFF) check --fix src/ api/ tests/
	$(RUFF) format src/ api/ tests/

run-api:
	$(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

dvc-push:
	AWS_ACCESS_KEY_ID=$${AWS_ACCESS_KEY_ID:-minioadmin} \
	AWS_SECRET_ACCESS_KEY=$${AWS_SECRET_ACCESS_KEY:-minioadmin} \
	AWS_DEFAULT_REGION=$${AWS_DEFAULT_REGION:-us-east-1} \
	$(DVC) push

dvc-pull:
	AWS_ACCESS_KEY_ID=$${AWS_ACCESS_KEY_ID:-minioadmin} \
	AWS_SECRET_ACCESS_KEY=$${AWS_SECRET_ACCESS_KEY:-minioadmin} \
	AWS_DEFAULT_REGION=$${AWS_DEFAULT_REGION:-us-east-1} \
	$(DVC) pull

push-model:
	AWS_ACCESS_KEY_ID=$${AWS_ACCESS_KEY_ID:-minioadmin} \
	AWS_SECRET_ACCESS_KEY=$${AWS_SECRET_ACCESS_KEY:-minioadmin} \
	AWS_DEFAULT_REGION=$${AWS_DEFAULT_REGION:-us-east-1} \
	$(PYTHON) scripts/push_model_minio.py --models-dir models --bucket $${MINIO_MLFLOW_BUCKET:-mlflow-artifacts} --prefix model

predict-test:
	$(PYTHON) scripts/test_prediction.py

clean:
	rm -rf .pytest_cache htmlcov .coverage .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f mlflow.db

