"""
Pydantic Schemas for FastAPI House Price Prediction Service
Defines strict input validation, serialization models, and OpenAPI examples.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HouseFeatures(BaseModel):
    """Features required to predict the price of a single residential property."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "area": 2200,
                "bedrooms": 3,
                "bathrooms": 2.5,
                "location": "suburban",
                "age": 12,
                "garage": 2,
            }
        }
    )

    area: float = Field(
        ...,
        ge=300,
        le=10000,
        description="Total living area in square feet (between 300 and 10,000 sqft)",
    )
    bedrooms: int = Field(
        ..., ge=1, le=10, description="Number of bedrooms (integer between 1 and 10)"
    )
    bathrooms: float = Field(
        ..., ge=1.0, le=8.0, description="Number of bathrooms (e.g. 1.0, 2.5, up to 8.0)"
    )
    location: Literal["urban", "suburban", "rural"] = Field(
        ..., description="Geographic classification: 'urban', 'suburban', or 'rural'"
    )
    age: int = Field(..., ge=0, le=100, description="Age of property in years (0 = brand new)")
    garage: int = Field(
        ..., ge=0, le=5, description="Garage capacity measured in number of vehicles (0 to 5)"
    )


class BatchHouseFeatures(BaseModel):
    """Batch input schema for scoring multiple properties."""

    houses: list[HouseFeatures] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    """Output prediction schema."""

    predicted_price: float = Field(..., description="Estimated market value in USD")
    currency: str = Field(default="USD", description="Currency symbol")
    model_name: str = Field(..., description="Identifier of the serving ML model")
    model_version: str = Field(..., description="Version of the serving model")
    features_received: dict[str, Any] = Field(..., description="Echo of input features scored")


class BatchPredictionResponse(BaseModel):
    """Output for batch prediction requests."""

    predictions: list[PredictionResponse]
    count: int
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    """Health status and service metadata."""

    status: str = Field(..., json_schema_extra={"example": "healthy"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})
    model_name: str = Field(..., json_schema_extra={"example": "house-price-predictor"})
    model_version: str = Field(..., json_schema_extra={"example": "1"})
    uptime_seconds: float = Field(..., json_schema_extra={"example": 124.5})


class DriftAnalysisResponse(BaseModel):
    """Output schema for statistical data drift analysis."""

    overall_drift_detected: bool
    drifted_features_count: int
    drifted_features: list[str]
    batch_size: int
    feature_details: dict[str, Any]
