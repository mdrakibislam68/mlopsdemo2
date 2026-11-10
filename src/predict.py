"""Prediction module for loading the saved pipeline and scoring new inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.utils import ensure_dir, load_json, setup_logging

logger = setup_logging()

DEFAULT_MODEL_PATH = Path("models/model.pkl")
DEFAULT_METADATA_PATH = Path("models/model_metadata.json")
DEFAULT_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def load_model_and_metadata(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path | None = DEFAULT_METADATA_PATH,
) -> tuple[Pipeline, dict[str, Any]]:
    """Load the saved Scikit-learn pipeline and optional metadata JSON.

    Args:
        model_path: Path to the serialized model.pkl file.
        metadata_path: Optional path to model_metadata.json.

    Returns:
        Tuple of (fitted_pipeline, metadata_dict).

    Raises:
        FileNotFoundError: If model_path does not exist.
    """
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(
            f"Trained model file not found at '{model_file}'. "
            "Run 'python -m src.train' to train and save the model first."
        )

    pipeline: Pipeline = joblib.load(model_file)
    metadata: dict[str, Any] = {}
    if metadata_path is not None:
        meta_file = Path(metadata_path)
        if meta_file.is_file():
            metadata = load_json(meta_file)

    return pipeline, metadata


def _coerce_input_to_dataframe(
    input_data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert supported input types into a Pandas DataFrame."""
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()
    if isinstance(input_data, dict):
        if not input_data:
            raise ValueError("Input dictionary cannot be empty.")
        return pd.DataFrame([input_data])
    if isinstance(input_data, list):
        if not input_data:
            raise ValueError("Input record list cannot be empty.")
        if not all(isinstance(item, dict) for item in input_data):
            raise ValueError("All elements in input list must be dictionaries.")
        return pd.DataFrame(input_data)
    raise TypeError(
        f"Unsupported input_data type '{type(input_data).__name__}'. "
        "Expected DataFrame, dict, or list of dicts."
    )


def validate_input_features(
    input_df: pd.DataFrame,
    expected_features: list[str],
) -> pd.DataFrame:
    """Validate and order input features before passing to the model pipeline.

    Args:
        input_df: Raw input DataFrame.
        expected_features: Ordered list of required feature column names.

    Returns:
        Validated DataFrame containing only expected_features in exact order,
        cast to float64.

    Raises:
        ValueError: If input is empty, missing required columns, contains non-numeric
            values, or contains infinite values.
    """
    if input_df.empty:
        raise ValueError("Input data cannot be empty.")

    missing_cols = [col for col in expected_features if col not in input_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    validated = input_df[expected_features].copy()

    for col in expected_features:
        if pd.api.types.is_bool_dtype(validated[col]):
            raise ValueError(f"Feature '{col}' contains boolean values; expected numeric values.")
        try:
            validated[col] = pd.to_numeric(validated[col], errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Feature '{col}' contains non-numeric values: {exc}") from exc

        if np.isinf(validated[col].to_numpy()).any():
            raise ValueError(f"Feature '{col}' contains infinite values (+/-inf).")

    return validated


def predict_house_prices(
    input_data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path | None = DEFAULT_METADATA_PATH,
) -> pd.DataFrame:
    """Validate input data and generate readable house price predictions.

    Args:
        input_data: Single record (dict), batch (list of dicts), or DataFrame.
        model_path: Path to saved model.pkl.
        metadata_path: Optional path to model_metadata.json.

    Returns:
        DataFrame with input features, predicted_med_house_val (in $100,000s),
        and predicted_price_usd (in USD).
    """
    pipeline, metadata = load_model_and_metadata(model_path=model_path, metadata_path=metadata_path)
    expected_features = metadata.get("feature_names", DEFAULT_FEATURES)

    raw_df = _coerce_input_to_dataframe(input_data)
    validated_df = validate_input_features(raw_df, expected_features=expected_features)

    predictions = pipeline.predict(validated_df)
    results_df = validated_df.copy()
    results_df["predicted_med_house_val"] = np.round(predictions, 4)
    results_df["predicted_price_usd"] = np.round(predictions * 100_000.0, 2)
    return results_df


def predict_from_csv(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path | None = DEFAULT_METADATA_PATH,
) -> pd.DataFrame:
    """Load samples from a CSV file, generate predictions, and optionally save to CSV.

    Args:
        input_csv: Path to input CSV containing feature columns.
        output_csv: Optional destination path for prediction output CSV.
        model_path: Path to saved model.pkl.
        metadata_path: Optional path to model_metadata.json.

    Returns:
        DataFrame containing features and prediction columns.

    Raises:
        FileNotFoundError: If input_csv does not exist.
        ValueError: If input CSV is empty or fails feature validation.
    """
    csv_path = Path(input_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")

    try:
        input_df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Input CSV file '{csv_path}' is empty.") from exc

    results_df = predict_house_prices(
        input_data=input_df,
        model_path=model_path,
        metadata_path=metadata_path,
    )

    if output_csv is not None:
        out_path = Path(output_csv)
        ensure_dir(out_path.parent)
        results_df.to_csv(out_path, index=False)
        logger.info("Saved %d predictions to %s", len(results_df), out_path)

    return results_df


def main() -> None:
    """CLI entry point for single JSON record or CSV batch prediction."""
    parser = argparse.ArgumentParser(description="Generate house price predictions from saved model.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to model.pkl.")
    parser.add_argument(
        "--metadata-path",
        default=str(DEFAULT_METADATA_PATH),
        help="Path to model_metadata.json.",
    )
    parser.add_argument("--input-csv", help="Path to input CSV file for batch prediction.")
    parser.add_argument("--output-csv", help="Optional path to save prediction results CSV.")
    parser.add_argument(
        "--input-json",
        help="JSON string representing a single feature dict or list of feature dicts.",
    )
    args = parser.parse_args()

    if args.input_csv:
        results = predict_from_csv(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            model_path=args.model_path,
            metadata_path=args.metadata_path,
        )
    elif args.input_json:
        payload = json.loads(args.input_json)
        results = predict_house_prices(
            input_data=payload,
            model_path=args.model_path,
            metadata_path=args.metadata_path,
        )
    else:
        parser.error("Provide either --input-csv or --input-json.")
        return

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

