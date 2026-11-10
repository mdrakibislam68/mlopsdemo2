"""
FastAPI Model Serving Application for MLOps Lab
Exposes REST endpoints for prediction, health check, batch scoring,
data drift detection, and Prometheus metrics scraping.
"""

import os
import sys
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.dependencies import ModelManager, get_model_manager
from api.schemas import (
    BatchHouseFeatures,
    BatchPredictionResponse,
    DriftAnalysisResponse,
    HealthResponse,
    HouseFeatures,
    PredictionResponse,
)
from src.monitoring.drift_detector import DriftDetector

# ---------------------------------------------------------
# Prometheus Metrics Definitions
# ---------------------------------------------------------
PREDICTION_REQUESTS_TOTAL = Counter(
    "mlops_prediction_requests_total",
    "Total number of prediction requests received",
    ["model_name", "status"],
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "mlops_prediction_latency_seconds",
    "Histogram of inference latency in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

PREDICTED_PRICE_USD = Histogram(
    "mlops_predicted_price_usd",
    "Distribution of predicted house prices in USD",
    buckets=[100000, 250000, 400000, 600000, 800000, 1000000, 1500000, 2000000],
)

INVALID_REQUESTS_TOTAL = Counter(
    "mlops_invalid_requests_total",
    "Count of invalid input requests failing schema validation",
    ["error_type"],
)

API_UPTIME_SECONDS = Gauge("mlops_api_uptime_seconds", "Time since API startup in seconds")

APP_START_TIME = time.time()
drift_detector_instance: DriftDetector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: loads model and drift baseline on startup."""
    global drift_detector_instance
    print("[FastAPI] Initializing House Price Prediction Service...")
    mm = get_model_manager()
    mm.load_model()

    # Initialize drift detector if reference data exists
    ref_path = os.getenv("REFERENCE_DATA_PATH", "data/raw/housing.csv")
    if os.path.exists(ref_path):
        try:
            drift_detector_instance = DriftDetector(reference_data_path=ref_path)
            print(f"[FastAPI] Drift detector initialized with reference dataset '{ref_path}'.")
        except Exception as e:
            print(f"[FastAPI] Note: Drift detector warning: {e}")
    else:
        print(
            f"[FastAPI] Notice: Reference dataset '{ref_path}' not found. Drift endpoint will be disabled."
        )

    yield
    print("[FastAPI] Shutting down service...")


app = FastAPI(
    title="House Price Prediction Service - MLOps Lab",
    description="Production-ready REST API for house price regression model serving, real-time monitoring, and data drift detection.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------
# Custom Validation Exception Handler
# ---------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Intercept validation failures to increment Prometheus metrics."""
    INVALID_REQUESTS_TOTAL.labels(error_type="pydantic_schema_validation").inc()
    PREDICTION_REQUESTS_TOTAL.labels(
        model_name="house-price-predictor", status="error_invalid_input"
    ).inc()
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid Input Data",
            "details": exc.errors(),
            "status": "validation_failed",
        },
    )


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@app.get("/", tags=["Info"])
async def root():
    """Service landing page redirecting to Swagger UI."""
    return {
        "service": "House Price Prediction API",
        "documentation": "/docs",
        "metrics": "/metrics",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(mm: ModelManager = Depends(get_model_manager)):
    """Health check endpoint for Docker orchestrators and Kubernetes probes."""
    uptime = time.time() - APP_START_TIME
    API_UPTIME_SECONDS.set(uptime)

    is_ready = mm.model is not None
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        model_name=mm.model_name,
        model_version=mm.model_version,
        uptime_seconds=round(uptime, 2),
    )


@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    """Prometheus metrics scraping endpoint."""
    API_UPTIME_SECONDS.set(time.time() - APP_START_TIME)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_single(features: HouseFeatures, mm: ModelManager = Depends(get_model_manager)):
    """
    Predict market value for a single house.
    Accepts validated numerical and categorical features.
    """
    start_time = time.perf_counter()

    if mm.model is None:
        PREDICTION_REQUESTS_TOTAL.labels(model_name=mm.model_name, status="error_no_model").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded on server."
        )

    try:
        # Construct DataFrame to maintain column order and feature names
        input_data = features.model_dump()
        df_input = pd.DataFrame([input_data])

        preds = mm.predict(df_input)
        price_pred = round(preds[0], 2)

        # Record metrics
        latency = time.perf_counter() - start_time
        PREDICTION_LATENCY_SECONDS.labels(endpoint="/predict").observe(latency)
        PREDICTED_PRICE_USD.observe(price_pred)
        PREDICTION_REQUESTS_TOTAL.labels(model_name=mm.model_name, status="success").inc()

        return PredictionResponse(
            predicted_price=price_pred,
            currency="USD",
            model_name=mm.model_name,
            model_version=mm.model_version,
            features_received=input_data,
        )

    except Exception as e:
        PREDICTION_REQUESTS_TOTAL.labels(model_name=mm.model_name, status="error_inference").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(batch: BatchHouseFeatures, mm: ModelManager = Depends(get_model_manager)):
    """
    Score a batch of multiple house feature records simultaneously.
    """
    start_time = time.perf_counter()

    if mm.model is None:
        PREDICTION_REQUESTS_TOTAL.labels(model_name=mm.model_name, status="error_no_model").inc()
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        data_records = [h.model_dump() for h in batch.houses]
        df_batch = pd.DataFrame(data_records)

        preds = mm.predict(df_batch)

        responses = []
        for feat, p in zip(data_records, preds):
            price_pred = round(p, 2)
            PREDICTED_PRICE_USD.observe(price_pred)
            responses.append(
                PredictionResponse(
                    predicted_price=price_pred,
                    currency="USD",
                    model_name=mm.model_name,
                    model_version=mm.model_version,
                    features_received=feat,
                )
            )

        latency = time.perf_counter() - start_time
        PREDICTION_LATENCY_SECONDS.labels(endpoint="/predict/batch").observe(latency)
        PREDICTION_REQUESTS_TOTAL.labels(model_name=mm.model_name, status="success").inc(
            len(responses)
        )

        return BatchPredictionResponse(
            predictions=responses,
            count=len(responses),
            model_name=mm.model_name,
            model_version=mm.model_version,
        )

    except Exception as e:
        PREDICTION_REQUESTS_TOTAL.labels(model_name=mm.model_name, status="error_batch").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/monitor/drift", response_model=DriftAnalysisResponse, tags=["Monitoring"])
async def analyze_drift(batch: BatchHouseFeatures):
    """
    Evaluate feature distribution of an incoming batch against the reference
    training dataset using Kolmogorov-Smirnov and Chi-Square statistical tests.
    """
    global drift_detector_instance
    if drift_detector_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Drift detector is not initialized. Reference baseline data is missing.",
        )

    data_records = [h.model_dump() for h in batch.houses]
    df_batch = pd.DataFrame(data_records)

    report = drift_detector_instance.analyze_batch(df_batch)
    return DriftAnalysisResponse(**report)
