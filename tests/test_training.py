"""Tests for model training, evaluation, selection, and artifact saving."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    compute_regression_metrics,
    evaluate_models,
    explain_best_model,
    format_comparison_table,
    select_best_model,
)
from src.train import build_candidate_estimators, run_training_pipeline, train_candidate_pipelines
from src.utils import load_config


def test_compute_regression_metrics() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])

    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics == {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "r2": 1.0}


def test_compute_regression_metrics_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_regression_metrics(np.array([]), np.array([]))

    with pytest.raises(ValueError, match="Length mismatch"):
        compute_regression_metrics(np.array([1.0, 2.0]), np.array([1.0]))


def test_build_candidate_estimators_requires_three_models() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        build_candidate_estimators({"linear_regression": {}})


def test_train_evaluate_and_select_best_model() -> None:
    rng = np.random.default_rng(42)
    n_samples = 120
    X = pd.DataFrame(
        {
            "MedInc": rng.uniform(1.0, 10.0, n_samples),
            "HouseAge": rng.uniform(1.0, 50.0, n_samples),
            "AveRooms": rng.uniform(3.0, 8.0, n_samples),
            "AveBedrms": rng.uniform(0.8, 1.5, n_samples),
            "Population": rng.uniform(100.0, 3000.0, n_samples),
            "AveOccup": rng.uniform(1.5, 4.5, n_samples),
            "Latitude": rng.uniform(33.0, 38.0, n_samples),
            "Longitude": rng.uniform(-122.0, -117.0, n_samples),
        }
    )
    y = pd.Series(0.5 * X["MedInc"] + 0.02 * X["HouseAge"] + rng.normal(0, 0.1, n_samples))

    models_cfg = {
        "linear_regression": {"fit_intercept": True},
        "decision_tree": {"max_depth": 5, "random_state": 42},
        "random_forest": {"n_estimators": 10, "max_depth": 6, "random_state": 42},
    }
    prep_cfg = {"imputer_strategy": "median", "scale_features": True}

    trained = train_candidate_pipelines(
        X_train=X.iloc[:90],
        y_train=y.iloc[:90],
        feature_names=list(X.columns),
        models_config=models_cfg,
        preprocessing_config=prep_cfg,
    )
    assert set(trained.keys()) == {"linear_regression", "decision_tree", "random_forest"}

    results_df = evaluate_models(trained, X.iloc[90:], y.iloc[90:])
    assert list(results_df.columns) == ["model", "mae", "mse", "rmse", "r2"]
    assert len(results_df) == 3
    assert results_df["rmse"].is_monotonic_increasing

    best_name, best_metrics = select_best_model(results_df, primary_metric="rmse")
    assert best_name == results_df.iloc[0]["model"]
    assert set(best_metrics.keys()) == {"mae", "mse", "rmse", "r2"}

    table = format_comparison_table(results_df)
    explanation = explain_best_model(best_name, results_df)
    assert best_name in table
    assert best_name in explanation


def test_end_to_end_training_saves_and_loads_artifacts(tmp_path: Path) -> None:
    config = load_config("config.yaml")
    # Speed up Random Forest in unit test while exercising real California Housing pipeline
    test_config_path = tmp_path / "test_config.yaml"
    config["models"]["random_forest"]["n_estimators"] = 10
    config["models"]["random_forest"]["max_depth"] = 8
    import yaml

    test_config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    output_dir = tmp_path / "models"
    result = run_training_pipeline(config_path=test_config_path, output_dir=output_dir)

    model_path = Path(result["model_path"])
    metadata_path = Path(result["metadata_path"])

    assert model_path.is_file()
    assert metadata_path.is_file()

    loaded_pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["best_model_name"] == result["best_model_name"]
    assert metadata["feature_names"] == config["dataset"]["feature_columns"]
    assert "training_configuration" in metadata
    assert "evaluation_metrics" in metadata
    assert len(metadata["evaluation_metrics"]["all_models"]) == 3

    sample_row = pd.DataFrame(
        [
            {
                "MedInc": 5.0,
                "HouseAge": 25.0,
                "AveRooms": 5.5,
                "AveBedrms": 1.05,
                "Population": 1200.0,
                "AveOccup": 2.8,
                "Latitude": 34.2,
                "Longitude": -118.4,
            }
        ]
    )
    preds = loaded_pipeline.predict(sample_row)
    assert preds.shape == (1,)
    assert np.isfinite(preds[0])

