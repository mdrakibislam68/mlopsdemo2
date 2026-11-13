.PHONY: help setup data validate preprocess train test lint run-api docker-up docker-down docker-build dvc-push dvc-pull push-model predict-test k8s-up k8s-down k8s-status k8s-verify argocd-up argocd-status argocd-password argocd-ui kserve-up kserve-status kserve-forward kserve-ui kfp-up kfp-status kfp-ui kfp-compile kfp-run clean

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
	@echo "  make k8s-up        - Create Kind cluster, load images, and deploy K8s stack"
	@echo "  make k8s-status    - Show Kind Kubernetes pods, services, and PVCs"
	@echo "  make k8s-verify    - Verify all Kind Kubernetes NodePorts & live inference"
	@echo "  make k8s-down      - Delete the Kind Kubernetes cluster"
	@echo "  make argocd-up     - Install Argo CD on Kind and apply k8s/argocd/application.yaml"
	@echo "  make argocd-status - Show Argo CD pods and Application sync status"
	@echo "  make argocd-password - Print initial Argo CD admin password"
	@echo "  make argocd-ui     - Port-forward Argo CD UI to https://localhost:8081"
	@echo "  make kserve-up     - Install KServe (Standard mode) & deploy InferenceService"
	@echo "  make kserve-status - Show KServe InferenceService and predictor pods"
	@echo "  make kserve-forward - Port-forward KServe predictor to http://localhost:8082"
	@echo "  make kserve-ui     - Port-forward Kubeflow KServe Models Web UI to http://localhost:8085"
	@echo "  make kfp-up        - Install Standalone Kubeflow Pipelines (KFP Lite) on Kind"
	@echo "  make kfp-status    - Show Kubeflow Pipelines pods and services"
	@echo "  make kfp-ui        - Port-forward Kubeflow Pipelines Dashboard to http://localhost:8080"
	@echo "  make kfp-compile   - Compile Python pipeline definition to YAML"
	@echo "  make kfp-run       - Upload pipeline and trigger an experiment run in KFP"
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

k8s-up:
	bash scripts/k8s_deploy.sh

k8s-status:
	kubectl get nodes -o wide
	kubectl get pods,svc,pvc -n mlops-lab -o wide

k8s-verify:
	$(PYTHON) scripts/verify_k8s.py

k8s-down:
	kind delete cluster --name mlops-lab

argocd-up:
	bash scripts/argocd_setup.sh

argocd-status:
	kubectl get pods,applications.argoproj.io -n argocd

argocd-password:
	@kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d; echo ""

argocd-ui:
	kubectl port-forward svc/argocd-server -n argocd 8081:443

kserve-up:
	bash scripts/kserve_setup.sh

kserve-status:
	kubectl get inferenceservice,pods -n mlops-lab

kserve-forward:
	kubectl port-forward svc/house-price-kserve-predictor -n mlops-lab 8082:80

kserve-ui:
	kubectl port-forward svc/kserve-models-web-application -n kserve 8085:80

kfp-up:
	bash scripts/kfp_setup.sh

kfp-status:
	kubectl get pods,svc -n kubeflow

kfp-ui:
	kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8080:80

kfp-compile:
	$(PYTHON) pipelines/house_price_pipeline.py

kfp-run:
	$(PYTHON) pipelines/run_pipeline.py

clean:
	rm -rf .pytest_cache htmlcov .coverage .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f mlflow.db

