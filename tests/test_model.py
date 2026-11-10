"""
Unit tests for model training, metrics calculation, and evaluation.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.evaluation.evaluate import compute_regression_metrics


def test_compute_regression_metrics_perfect():
    """Verify metrics calculation for perfect predictions."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])

    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2_score"] == 1.0


def test_compute_regression_metrics_known_error():
    """Verify metrics calculation against manual calculation."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])  # Errors: +10, -10, +10

    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] == 10.0
    assert metrics["mse"] == 100.0
    assert metrics["rmse"] == 10.0
    assert metrics["r2_score"] < 1.0


def test_linear_regression_training():
    """Verify Linear Regression fits and produces reasonable predictions."""
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = X[:, 0] * 50 + X[:, 1] * 20 + 10

    lr = LinearRegression()
    lr.fit(X, y)
    preds = lr.predict(X)

    metrics = compute_regression_metrics(y, preds)
    assert metrics["r2_score"] > 0.95
    assert len(preds) == 100


def test_random_forest_training():
    """Verify Random Forest fits and produces reasonable predictions."""
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = X[:, 0] * 50 + (X[:, 1] ** 2) * 20 + 10

    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X, y)
    preds = rf.predict(X)

    metrics = compute_regression_metrics(y, preds)
    assert metrics["r2_score"] > 0.80
    assert len(preds) == 100
