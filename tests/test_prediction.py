"""Tests for prediction module, CSV inference, and invalid input handling."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.predict import (
    DEFAULT_FEATURES,
    load_model_and_metadata,
    predict_from_csv,
    predict_house_prices,
    validate_input_features,
)
from src.preprocessing import build_preprocessor


@pytest.fixture
def trained_artifacts(tmp_path: Path, sample_housing_df: pd.DataFrame) -> tuple[Path, Path]:
    """Create a fitted pipeline and metadata JSON inside a temp directory."""
    X = sample_housing_df[DEFAULT_FEATURES]
    y = sample_housing_df["MedHouseVal"]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(DEFAULT_FEATURES)),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(X, y)

    model_path = tmp_path / "model.pkl"
    metadata_path = tmp_path / "model_metadata.json"
    joblib.dump(pipeline, model_path)
    metadata_path.write_text(
        json.dumps({"best_model_name": "linear_regression", "feature_names": DEFAULT_FEATURES}),
        encoding="utf-8",
    )
    return model_path, metadata_path


def test_predict_single_dict_and_batch_list(trained_artifacts: tuple[Path, Path]) -> None:
    model_path, metadata_path = trained_artifacts
    sample = {
        "MedInc": 6.5,
        "HouseAge": 30.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.0,
        "Population": 800.0,
        "AveOccup": 2.5,
        "Latitude": 37.85,
        "Longitude": -122.25,
    }

    single_result = predict_house_prices(sample, model_path=model_path, metadata_path=metadata_path)
    assert len(single_result) == 1
    assert "predicted_med_house_val" in single_result.columns
    assert "predicted_price_usd" in single_result.columns
    assert np.isclose(
        single_result["predicted_price_usd"].iloc[0],
        single_result["predicted_med_house_val"].iloc[0] * 100_000.0,
        atol=10.0,
    )

    batch_result = predict_house_prices(
        [sample, sample], model_path=model_path, metadata_path=metadata_path
    )
    assert len(batch_result) == 2


def test_predict_handles_missing_value_via_imputer(trained_artifacts: tuple[Path, Path]) -> None:
    model_path, metadata_path = trained_artifacts
    sample_with_nan = {
        "MedInc": np.nan,
        "HouseAge": 30.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.0,
        "Population": 800.0,
        "AveOccup": 2.5,
        "Latitude": 37.85,
        "Longitude": -122.25,
    }

    result = predict_house_prices(
        sample_with_nan, model_path=model_path, metadata_path=metadata_path
    )
    assert len(result) == 1
    assert np.isfinite(result["predicted_med_house_val"].iloc[0])


def test_predict_from_csv(
    tmp_path: Path,
    trained_artifacts: tuple[Path, Path],
    sample_housing_df: pd.DataFrame,
) -> None:
    model_path, metadata_path = trained_artifacts
    input_csv = tmp_path / "inputs.csv"
    output_csv = tmp_path / "outputs.csv"

    sample_housing_df[DEFAULT_FEATURES].to_csv(input_csv, index=False)

    results = predict_from_csv(
        input_csv=input_csv,
        output_csv=output_csv,
        model_path=model_path,
        metadata_path=metadata_path,
    )
    assert len(results) == len(sample_housing_df)
    assert output_csv.is_file()
    saved_df = pd.read_csv(output_csv)
    assert "predicted_price_usd" in saved_df.columns


def test_invalid_inputs_raise_clear_errors(
    tmp_path: Path,
    trained_artifacts: tuple[Path, Path],
) -> None:
    model_path, metadata_path = trained_artifacts

    # Missing model file
    with pytest.raises(FileNotFoundError, match="Trained model file not found"):
        load_model_and_metadata(tmp_path / "missing.pkl")

    # Missing CSV file
    with pytest.raises(FileNotFoundError, match="Input CSV file not found"):
        predict_from_csv(tmp_path / "missing.csv", model_path=model_path, metadata_path=metadata_path)

    # Empty dict / list / DataFrame
    with pytest.raises(ValueError, match="empty"):
        predict_house_prices({}, model_path=model_path, metadata_path=metadata_path)

    with pytest.raises(ValueError, match="empty"):
        predict_house_prices([], model_path=model_path, metadata_path=metadata_path)

    with pytest.raises(ValueError, match="empty"):
        predict_house_prices(pd.DataFrame(), model_path=model_path, metadata_path=metadata_path)

    # Unsupported type
    with pytest.raises(TypeError, match="Unsupported input_data type"):
        predict_house_prices("invalid_string", model_path=model_path, metadata_path=metadata_path)  # type: ignore[arg-type]

    # Missing required columns
    with pytest.raises(ValueError, match="Missing required feature columns"):
        validate_input_features(pd.DataFrame([{"MedInc": 5.0}]), DEFAULT_FEATURES)

    # Non-numeric string values
    bad_df = pd.DataFrame(
        [
            {
                "MedInc": "not_a_number",
                "HouseAge": 30.0,
                "AveRooms": 6.0,
                "AveBedrms": 1.0,
                "Population": 800.0,
                "AveOccup": 2.5,
                "Latitude": 37.85,
                "Longitude": -122.25,
            }
        ]
    )
    with pytest.raises(ValueError, match="non-numeric"):
        validate_input_features(bad_df, DEFAULT_FEATURES)

    # Boolean values
    bool_df = bad_df.copy()
    bool_df["MedInc"] = True
    with pytest.raises(ValueError, match="boolean"):
        validate_input_features(bool_df, DEFAULT_FEATURES)

    # Infinite values
    inf_df = bad_df.copy()
    inf_df["MedInc"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        validate_input_features(inf_df, DEFAULT_FEATURES)

