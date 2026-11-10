"""
Evaluation Metrics Module for MLOps Lab
Calculates regression evaluation metrics: MAE, MSE, RMSE, R2 Score.
"""

import json
from typing import Any

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculate MAE, MSE, RMSE, and R2 score.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
    }


def save_metrics_to_json(metrics: dict[str, Any], filepath: str) -> None:
    """Save metrics dictionary to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
