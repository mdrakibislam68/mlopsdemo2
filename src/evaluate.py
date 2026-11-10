"""Model evaluation, comparison, and best-model selection utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.data_loader import load_housing_dataset, split_features_and_target
from src.preprocessing import split_train_test
from src.utils import load_config, load_json, setup_logging

logger = setup_logging()

SUPPORTED_PRIMARY_METRICS = ("rmse", "mae", "mse", "r2")


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute standard regression evaluation metrics.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predicted target values.

    Returns:
        Dictionary with mae, mse, rmse, and r2 scores.

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        raise ValueError("y_true and y_pred must be non-empty.")
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true has {y_true_arr.shape[0]} samples, "
            f"y_pred has {y_pred_arr.shape[0]}."
        )

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    return {
        "mae": round(mae, 6),
        "mse": round(mse, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6),
    }


def evaluate_models(
    trained_models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate multiple trained regression pipelines on the test dataset.

    Args:
        trained_models: Mapping of model name to fitted Scikit-learn Pipeline.
        X_test: Held-out test features.
        y_test: Held-out test targets.

    Returns:
        DataFrame sorted by RMSE ascending with columns: model, mae, mse, rmse, r2.

    Raises:
        ValueError: If trained_models is empty or test data is empty.
    """
    if not trained_models:
        raise ValueError("trained_models must contain at least one fitted model.")
    if X_test.empty or y_test.empty:
        raise ValueError("X_test and y_test must be non-empty.")

    records: list[dict[str, Any]] = []
    for model_name, pipeline in trained_models.items():
        predictions = pipeline.predict(X_test)
        metrics = compute_regression_metrics(y_test, predictions)
        records.append({"model": model_name, **metrics})

    results_df = pd.DataFrame(records).sort_values(by="rmse", ascending=True).reset_index(drop=True)
    return results_df


def select_best_model(
    results_df: pd.DataFrame,
    primary_metric: str = "rmse",
) -> tuple[str, dict[str, float]]:
    """Select the best model from an evaluation results DataFrame.

    Args:
        results_df: DataFrame produced by evaluate_models.
        primary_metric: Metric used for selection ('rmse', 'mae', 'mse', or 'r2').

    Returns:
        Tuple of (best_model_name, metrics_dict).

    Raises:
        ValueError: If results_df is empty or primary_metric is invalid.
    """
    if results_df.empty:
        raise ValueError("Cannot select best model from an empty results table.")
    if primary_metric not in SUPPORTED_PRIMARY_METRICS:
        raise ValueError(
            f"Unsupported primary_metric '{primary_metric}'. "
            f"Supported: {SUPPORTED_PRIMARY_METRICS}."
        )
    if primary_metric not in results_df.columns:
        raise ValueError(f"Primary metric '{primary_metric}' not found in results columns.")

    ascending = primary_metric != "r2"
    sorted_df = results_df.sort_values(by=primary_metric, ascending=ascending).reset_index(drop=True)
    best_row = sorted_df.iloc[0]

    best_name = str(best_row["model"])
    best_metrics = {
        "mae": float(best_row["mae"]),
        "mse": float(best_row["mse"]),
        "rmse": float(best_row["rmse"]),
        "r2": float(best_row["r2"]),
    }
    return best_name, best_metrics


def format_comparison_table(results_df: pd.DataFrame) -> str:
    """Format model evaluation results into a clean readable ASCII table."""
    display_df = results_df.copy()
    display_df.columns = ["Model", "MAE", "MSE", "RMSE", "R2"]
    for col in ("MAE", "MSE", "RMSE", "R2"):
        display_df[col] = display_df[col].map(lambda val: f"{val:.4f}")

    border = "=" * 68
    table_str = display_df.to_string(index=False)
    return f"{border}\nMODEL EVALUATION COMPARISON (Sorted by Test RMSE)\n{border}\n{table_str}\n{border}"


def explain_best_model(best_model_name: str, results_df: pd.DataFrame) -> str:
    """Generate a clear technical explanation for why the selected model performed best."""
    best_row = results_df.loc[results_df["model"] == best_model_name].iloc[0]
    best_rmse = float(best_row["rmse"])
    best_r2 = float(best_row["r2"])
    best_mae = float(best_row["mae"])

    runners_up = results_df.loc[results_df["model"] != best_model_name]
    comparison_notes: list[str] = []
    for _, row in runners_up.iterrows():
        other_rmse = float(row["rmse"])
        diff = other_rmse - best_rmse
        pct = (diff / other_rmse * 100.0) if other_rmse > 0 else 0.0
        comparison_notes.append(
            f"  - Outperforms {row['model']} by {diff:.4f} RMSE ({pct:.1f}% lower error; "
            f"R2={row['r2']:.4f} vs {best_r2:.4f})."
        )

    rationale_by_model = {
        "random_forest": (
            "Random Forest Regressor averages predictions across an ensemble of decorrelated "
            "decision trees trained on bootstrap samples and random feature subsets. This reduces "
            "prediction variance compared to a single Decision Tree while capturing non-linear "
            "relationships and spatial feature interactions (e.g., Latitude, Longitude, MedInc) "
            "that Linear Regression cannot model."
        ),
        "decision_tree": (
            "Decision Tree Regressor captures non-linear thresholds and feature interactions "
            "across housing attributes without requiring global linearity assumptions."
        ),
        "linear_regression": (
            "Linear Regression models additive linear relationships with low variance and "
            "strong generalization when target relationships are predominantly linear."
        ),
    }

    specific_reason = rationale_by_model.get(
        best_model_name,
        f"{best_model_name} achieved the lowest test-set RMSE among all evaluated candidates.",
    )

    lines = [
        f"Selected Best Model: {best_model_name}",
        f"Primary Metric (Test RMSE): {best_rmse:.4f} (MAE: {best_mae:.4f}, R2: {best_r2:.4f})",
        "",
        "Why this model performs better:",
        f"  {specific_reason}",
    ]
    if comparison_notes:
        lines.append("")
        lines.append("Benchmark Comparison:")
        lines.extend(comparison_notes)
    return "\n".join(lines)


def evaluate_saved_model(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load the saved best model and evaluate it on the configured test split."""
    config = load_config(config_path)
    output_cfg = config["output"]
    dataset_cfg = config["dataset"]
    prep_cfg = config["preprocessing"]

    model_dir = Path(output_cfg["model_dir"])
    model_path = model_dir / output_cfg["model_filename"]
    metadata_path = model_dir / output_cfg["metadata_filename"]

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Saved model not found at {model_path}. Run 'python -m src.train' first."
        )

    pipeline: Pipeline = joblib.load(model_path)
    df, _, _ = load_housing_dataset(data_home=dataset_cfg.get("data_home"))
    target_col = dataset_cfg.get("target_column", "MedHouseVal")
    X, y = split_features_and_target(df, target_column=target_col)
    _, X_test, _, y_test = split_train_test(
        X,
        y,
        test_size=float(prep_cfg["test_size"]),
        random_state=int(prep_cfg["random_state"]),
    )

    y_pred = pipeline.predict(X_test)
    live_metrics = compute_regression_metrics(y_test, y_pred)

    if metadata_path.is_file():
        metadata = load_json(metadata_path)
        all_results = metadata.get("evaluation_metrics", {}).get("all_models", [])
        if all_results:
            results_df = pd.DataFrame(all_results).sort_values(by="rmse", ascending=True).reset_index(drop=True)
            print(format_comparison_table(results_df))
            print()
            best_name = metadata.get("best_model_name", "unknown")
            print(explain_best_model(best_name, results_df))
            print()

    logger.info(
        "Verified saved model '%s' on test split (%d samples): RMSE=%.4f, MAE=%.4f, MSE=%.4f, R2=%.4f",
        model_path,
        len(X_test),
        live_metrics["rmse"],
        live_metrics["mae"],
        live_metrics["mse"],
        live_metrics["r2"],
    )
    return live_metrics


def main() -> None:
    """CLI entry point for evaluating the saved model and displaying comparisons."""
    parser = argparse.ArgumentParser(description="Evaluate the trained house price prediction model.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    args = parser.parse_args()
    evaluate_saved_model(config_path=args.config)


if __name__ == "__main__":
    main()

