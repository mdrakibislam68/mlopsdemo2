"""Tests for data loading and exploratory data analysis."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import (
    format_eda_report,
    generate_eda_summary,
    load_housing_dataset,
    split_features_and_target,
)


def test_load_housing_dataset_shape_and_columns() -> None:
    df, feature_names, target_name = load_housing_dataset()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (20640, 9)
    assert target_name == "MedHouseVal"
    assert len(feature_names) == 8
    assert set(feature_names).issubset(df.columns)
    assert target_name in df.columns


def test_split_features_and_target(sample_housing_df: pd.DataFrame) -> None:
    X, y = split_features_and_target(sample_housing_df, target_column="MedHouseVal")

    assert X.shape == (len(sample_housing_df), 8)
    assert y.shape == (len(sample_housing_df),)
    assert "MedHouseVal" not in X.columns
    assert y.name == "MedHouseVal"


def test_split_features_and_target_invalid_inputs(sample_housing_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="empty"):
        split_features_and_target(pd.DataFrame())

    with pytest.raises(ValueError, match="Target column"):
        split_features_and_target(sample_housing_df, target_column="NonExistent")


def test_generate_eda_summary_and_report(sample_housing_df: pd.DataFrame) -> None:
    df_with_issues = pd.concat(
        [sample_housing_df, sample_housing_df.iloc[[0]]], ignore_index=True
    )
    df_with_issues.loc[1, "MedInc"] = None

    summary = generate_eda_summary(df_with_issues, target_column="MedHouseVal")

    assert summary["shape"] == (7, 9)
    assert summary["total_missing_values"] == 1
    assert summary["missing_values"]["MedInc"] == 1
    assert summary["duplicate_records"] == 1
    assert "MedInc" in summary["target_correlations"]

    report = format_eda_report(summary, target_column="MedHouseVal")
    assert "CALIFORNIA HOUSING DATASET" in report
    assert "Statistical Summary:" in report

