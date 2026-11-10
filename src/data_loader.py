"""Data loading and basic exploratory data analysis for California Housing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import fetch_california_housing

from src.utils import ensure_dir, load_config, setup_logging

logger = setup_logging()


def load_housing_dataset(
    data_home: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str], str]:
    """Load the California Housing dataset as a Pandas DataFrame.

    Args:
        data_home: Optional local cache directory passed to scikit-learn.

    Returns:
        Tuple of (full_dataframe, feature_names, target_name).

    Raises:
        RuntimeError: If loading the dataset fails or yields an empty frame.
    """
    try:
        dataset = fetch_california_housing(
            data_home=str(data_home) if data_home is not None else None,
            as_frame=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load California Housing dataset: {exc}") from exc

    df = dataset.frame.copy()
    if df.empty:
        raise RuntimeError("Loaded California Housing dataset is empty.")

    feature_names = list(dataset.feature_names)
    target_name = str(dataset.target_names[0])
    return df, feature_names, target_name


def split_features_and_target(
    df: pd.DataFrame,
    target_column: str = "MedHouseVal",
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate input features and target variable from a DataFrame.

    Args:
        df: Full dataset containing features and target.
        target_column: Name of the target column.

    Returns:
        Tuple of (features_df, target_series).

    Raises:
        ValueError: If df is empty or target_column is missing.
    """
    if df.empty:
        raise ValueError("Cannot split features and target from an empty DataFrame.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame columns: {list(df.columns)}")

    features = df.drop(columns=[target_column]).copy()
    target = df[target_column].copy()
    return features, target


def generate_eda_summary(
    df: pd.DataFrame,
    target_column: str = "MedHouseVal",
) -> dict[str, Any]:
    """Compute basic exploratory data analysis metrics for a DataFrame.

    Args:
        df: Dataset DataFrame.
        target_column: Target column name for correlation analysis.

    Returns:
        Dictionary containing shape, dtypes, missing values, duplicates,
        statistical summary, and target correlations.

    Raises:
        ValueError: If df is empty.
    """
    if df.empty:
        raise ValueError("Cannot generate EDA summary for an empty DataFrame.")

    missing_per_col = {col: int(count) for col, count in df.isna().sum().items()}
    total_missing = int(sum(missing_per_col.values()))
    duplicate_count = int(df.duplicated().sum())
    stats_df = df.describe().T[["count", "mean", "std", "min", "50%", "max"]]
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    target_correlations: dict[str, float] = {}
    if target_column in df.columns:
        numeric_df = df.select_dtypes(include="number")
        if target_column in numeric_df.columns:
            corr_series = numeric_df.corr()[target_column].drop(labels=[target_column], errors="ignore")
            target_correlations = {
                col: round(float(val), 4)
                for col, val in corr_series.sort_values(ascending=False).items()
            }

    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing_values": missing_per_col,
        "total_missing_values": total_missing,
        "duplicate_records": duplicate_count,
        "statistical_summary": stats_df,
        "target_correlations": target_correlations,
    }


def format_eda_report(summary: dict[str, Any], target_column: str = "MedHouseVal") -> str:
    """Format an EDA summary dictionary into a human-readable report."""
    rows, cols = summary["shape"]
    lines = [
        "=" * 70,
        "EXPLORATORY DATA ANALYSIS — CALIFORNIA HOUSING DATASET",
        "=" * 70,
        f"Dataset Shape      : {rows:,} rows x {cols} columns",
        f"Total Missing      : {summary['total_missing_values']}",
        f"Duplicate Records  : {summary['duplicate_records']}",
        "",
        "Columns & Data Types:",
    ]
    for col, dtype in summary["dtypes"].items():
        missing = summary["missing_values"].get(col, 0)
        lines.append(f"  - {col:<15} | dtype={dtype:<10} | missing={missing}")

    lines.extend(
        [
            "",
            "Statistical Summary:",
            summary["statistical_summary"].round(4).to_string(),
        ]
    )

    if summary["target_correlations"]:
        lines.extend(["", f"Feature Correlations with '{target_column}':"])
        for col, corr in summary["target_correlations"].items():
            lines.append(f"  - {col:<15} : {corr:+.4f}")

    lines.append("=" * 70)
    return "\n".join(lines)


def run_data_loading(config_path: str | Path = "config.yaml", export_csv: bool = False) -> pd.DataFrame:
    """Load dataset, run EDA report, and optionally export raw CSV."""
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    target_col = dataset_cfg.get("target_column", "MedHouseVal")
    data_home = dataset_cfg.get("data_home")

    df, feature_names, target_name = load_housing_dataset(data_home=data_home)
    features, target = split_features_and_target(df, target_column=target_col)

    logger.info(
        "Loaded dataset with %d samples, %d features (%s), target='%s'.",
        len(df),
        len(feature_names),
        ", ".join(feature_names),
        target_name,
    )
    logger.info("Separated X shape=%s, y shape=%s.", features.shape, target.shape)

    summary = generate_eda_summary(df, target_column=target_col)
    report = format_eda_report(summary, target_column=target_col)
    print(report)

    if export_csv:
        export_path = Path(dataset_cfg.get("export_csv_path", "data/california_housing.csv"))
        ensure_dir(export_path.parent)
        df.to_csv(export_path, index=False)
        logger.info("Exported raw dataset CSV to %s", export_path)

    return df


def main() -> None:
    """CLI entry point for data loading and exploratory analysis."""
    parser = argparse.ArgumentParser(description="Load and inspect the California Housing dataset.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Optionally export the loaded dataset to the CSV path configured in config.yaml.",
    )
    args = parser.parse_args()
    run_data_loading(config_path=args.config, export_csv=args.export_csv)


if __name__ == "__main__":
    main()

