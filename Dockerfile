# Multi-stage production-ready Dockerfile for FastAPI Prediction Service
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install serving dependencies into /install so non-root runtime user can access them
RUN pip install --no-cache-dir --prefix=/install \
    "numpy>=1.24.0" \
    "pandas>=2.0.0" \
    "scikit-learn>=1.3.0" \
    "scipy>=1.10.0" \
    "joblib>=1.3.0" \
    "pydantic>=2.5.0" \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.28.0" \
    "prometheus-client>=0.20.0" \
    "mlflow-skinny>=3.0.0" \
    "boto3>=1.34.0" \
    "psycopg2-binary>=2.9.9" \
    "pyyaml>=6.0.0"

# Final runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

# Install curl for container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages into /usr/local
COPY --from=builder /install /usr/local

# Create non-root application user
RUN useradd -m -u 1001 appuser && \
    mkdir -p /app/models /app/data/raw && \
    chown -R appuser:appuser /app

# Copy application code, configuration, models and baseline reference data
COPY --chown=appuser:appuser params.yaml /app/
COPY --chown=appuser:appuser api/ /app/api/
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser models/ /app/models/
COPY --chown=appuser:appuser data/raw/housing.csv /app/data/raw/housing.csv

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_LOCAL_PATH=/app/models/full_pipeline.joblib \
    REFERENCE_DATA_PATH=/app/data/raw/housing.csv

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
