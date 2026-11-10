# MLflow Tracking Server & Metadata Architecture

## 1. What Problem Does MLflow Solve?

Without an experiment tracking framework, machine learning experimentation suffers from severe challenges:
- Hyperparameters, code versions, and evaluation scores are lost across Jupyter notebooks or terminal outputs.
- Team members cannot reproduce why model A achieved lower error than model B.
- No central audit trail connects deployed model binaries to the dataset and Git commit that produced them.

**MLflow** solves this by providing:
1. **MLflow Tracking:** Centralized logging of parameters, metrics, code versions, tags, and output artifacts.
2. **MLflow Models:** A standardized packaging format ensuring models can be deployed across Docker, Kubernetes, or serverless functions with identical behavior.
3. **MLflow Model Registry:** A collaborative hub for tracking model versions, reviewing lineage, and promoting models across lifecycle stages (`Staging`, `Production`, `@champion`, `@challenger`).

---

## 2. Why Are We Using MLflow in this Lab?

1. **Comparison of Competing Algorithms:** In `src/training/train.py`, we train both **Linear Regression** and **Random Forest Regressor**. MLflow logs train and test RMSE, MAE, MSE, and R² side-by-side.
2. **Automated Promotion:** Automatically registers the best-performing model into the MLflow Model Registry and tags it as `@champion`.
3. **Production Model Decoupling:** Downstream serving services (FastAPI) query MLflow directly to download and initialize the latest validated production model without hardcoding file paths.

---

## 3. How MLflow Works Internally

```mermaid
flowchart TD
    subgraph Client["Python Training Script / Client"]
        LogP["mlflow.log_param()"]
        LogM["mlflow.log_metric()"]
        LogMod["mlflow.sklearn.log_model()"]
    end

    subgraph Server["MLflow Tracking Server (:5001)"]
        RestAPI["REST API Router (/api/2.0/mlflow/...)"]
        SQLAlchemy["SQLAlchemy ORM Layer"]
        BotoAdapter["Boto3 S3 Artifact Adapter"]
    end

    subgraph BackendStore["Metadata Store (PostgreSQL)"]
        Tables["runs, experiments, params, metrics, model_versions"]
    end

    subgraph ArtifactStore["Artifact Store (MinIO)"]
        S3Buckets["s3://mlflow-artifacts/<run_id>/artifacts/"]
    end

    LogP -->|HTTP POST JSON| RestAPI
    LogM -->|HTTP POST JSON| RestAPI
    LogMod -->|HTTP POST Model| RestAPI

    RestAPI --> SQLAlchemy
    SQLAlchemy --> Tables

    RestAPI --> BotoAdapter
    BotoAdapter -->|S3 Multipart Upload| S3Buckets
```

### MLflow Server Command Breakdown
In `docker/Dockerfile.mlflow`, the tracking server is started with:
```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB} \
  --default-artifact-root s3://${MINIO_MLFLOW_BUCKET}/
```

- `--backend-store-uri`: Points to the relational database (PostgreSQL) storing metadata.
- `--default-artifact-root`: Points to the object storage URI (MinIO S3 bucket) where model binaries and plots are written.

---

## 4. Key Configuration Parameters

| Variable | Scope | Purpose |
| :--- | :--- | :--- |
| `MLFLOW_TRACKING_URI` | Python scripts / API | `http://localhost:5001` (host) or `http://mlflow:5000` (container network). |
| `MLFLOW_S3_ENDPOINT_URL` | MLflow & API containers | `http://minio:9000` tells boto3 to route S3 requests to MinIO instead of AWS. |
| `AWS_ACCESS_KEY_ID` | MLflow & API containers | MinIO authentication root user (`minioadmin`). |
| `AWS_SECRET_ACCESS_KEY` | MLflow & API containers | MinIO authentication password (`minioadmin`). |
| `MINIO_MLFLOW_BUCKET` | MLflow container | Default bucket name (`mlflow-artifacts`). |

---

## 5. Model Signatures and Input Examples

In `src/training/train.py`, we enforce strict MLflow Model Signatures:

```python
from mlflow.models.signature import infer_signature

input_sample = X_test.head(5)
output_sample = model.predict(input_sample)
signature = infer_signature(input_sample, output_sample)

mlflow.sklearn.log_model(
    sk_model=model,
    name="model",
    signature=signature,
    input_example=input_sample.iloc[[0]],
    serialization_format="cloudpickle"
)
```

**Why this matters:**
- **Schema Contract:** Records expected input feature names and dtypes. If an incoming inference request has mismatched types, MLflow flags it immediately.
- **Explainability:** Enables tools to inspect expected inputs directly from the MLflow UI without reading Python source code.

---

## 6. Accessing the MLflow UI

When Docker Compose is running:
- Open your browser at **`http://localhost:5001`**.
- View the active experiment **`house-price-prediction`**.
- Click into runs to view metric history graphs (Train vs Test RMSE), hyperparameters, and downloaded model artifacts.
- Navigate to the **Models** tab at the top to view `house-price-predictor`, its versions, and current `@champion` status.

---

## 7. Troubleshooting Common MLflow Errors

### Error: `mlflow.exceptions.MlflowException: API request to http://localhost:5001 failed`
- **Cause:** MLflow container is still starting up or PostgreSQL healthcheck hasn't completed.
- **Solution:** Check container status via `docker compose ps` and logs via `docker compose logs mlflow`.

### Error: `AirPlay Port 5000 Conflict (macOS)`
- **Symptom:** Port 5000 is occupied by `ControlCenter` on macOS Sonoma/Sequoia.
- **Solution:** In `.env`, we map host port `5001` to container port `5000` (`127.0.0.1:5001:5000`), completely bypassing macOS AirPlay conflicts.

### Error: `EndpointConnectionError: Could not connect to the endpoint URL: http://minio:9000`
- **Cause:** `MLFLOW_S3_ENDPOINT_URL` is missing or points to `localhost` from inside a container.
- **Solution:** Set `MLFLOW_S3_ENDPOINT_URL=http://minio:9000` inside container environments.

