"""
Data Validation Module for MLOps Lab
Ensures input datasets conform to the required schema, value ranges,
and data integrity rules prior to preprocessing or model training.
"""

import argparse
import os
import sys
from typing import Any

import pandas as pd
import yaml


class DataValidationError(Exception):
    """Custom exception raised when data validation fails."""

    pass


def validate_schema(df: pd.DataFrame, config: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate that the DataFrame satisfies column, type, and range constraints.
    Returns (is_valid, error_messages).
    """
    errors: list[str] = []
    val_cfg = config.get("data_validation", {})
    required_cols = val_cfg.get("required_columns", [])
    numerical_cols = val_cfg.get("numerical_columns", [])
    val_cfg.get("categorical_columns", [])
    ranges = val_cfg.get("ranges", {})

    # 1. Check for empty dataframe
    if df.empty:
        errors.append("Dataset is empty (0 rows).")
        return False, errors

    # 2. Check required columns
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # 3. Check numerical columns have numerical types
    for col in numerical_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' expected numeric, but got {df[col].dtype}.")

    # 4. Check categorical columns
    valid_locations = {"urban", "suburban", "rural"}
    if "location" in df.columns:
        invalid_locs = set(df["location"].dropna().unique()) - valid_locations
        if invalid_locs:
            errors.append(f"Column 'location' contains unexpected categories: {invalid_locs}")

    # 5. Check ranges
    for col, (min_val, max_val) in ranges.items():
        if col in df.columns:
            non_null_vals = df[col].dropna()
            if (non_null_vals < min_val).any():
                violators = (non_null_vals < min_val).sum()
                errors.append(f"Column '{col}' has {violators} values below minimum {min_val}.")
            if (non_null_vals > max_val).any():
                violators = (non_null_vals > max_val).sum()
                errors.append(f"Column '{col}' has {violators} values above maximum {max_val}.")

    is_valid = len(errors) == 0
    return is_valid, errors


def run_validation(data_path: str, config_path: str) -> bool:
    """Load dataset, validate against config, and report findings."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(data_path)
    print(
        f"[DataVal] Validating '{data_path}' with {len(df)} rows and {len(df.columns)} columns..."
    )

    is_valid, errors = validate_schema(df, config)

    if not is_valid:
        print("[DataVal] Validation FAILED with errors:")
        for err in errors:
            print(f"  - ERROR: {err}")
        return False

    # Check missing values stats
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        print("[DataVal] Notice: Missing values detected (will be imputed during preprocessing):")
        for c, count in null_cols.items():
            print(f"  - {c}: {count} missing ({count / len(df):.1%})")
    else:
        print("[DataVal] No missing values detected.")

    print(f"[DataVal] PASSED: Dataset '{data_path}' meets all schema integrity specifications.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate dataset integrity and schema")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--data", default=None, help="Path to CSV dataset")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_path = args.data or config["data_generation"]["raw_data_path"]

    try:
        success = run_validation(data_path=data_path, config_path=args.config)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"[DataVal] Fatal exception during validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
