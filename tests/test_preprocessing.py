"""Tests for train/test splitting and numerical preprocessing pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import split_features_and_target
from src.preprocessing import build_preprocessor, split_train_test


def test_split_train_test_reproducibility(sample_housing_df: pd.DataFrame) -> None:
    X, y = split_features_and_target(sample_housing_df)
    X_train_1, X_test_1, y_train_1, y_test_1 = split_train_test(
        X, y, test_size=0.33, random_state=42
    )
    X_train_2, X_test_2, y_train_2, y_test_2 = split_train_test(
        X, y, test_size=0.33, random_state=42
    )

    pd.testing.assert_frame_equal(X_train_1, X_train_2)
    pd.testing.assert_frame_equal(X_test_1, X_test_2)
    pd.testing.assert_series_equal(y_train_1, y_train_2)
    pd.testing.assert_series_equal(y_test_1, y_test_2)
    assert len(X_train_1) + len(X_test_1) == len(X)


def test_split_train_test_invalid_inputs(sample_housing_df: pd.DataFrame) -> None:
    X, y = split_features_and_target(sample_housing_df)

    with pytest.raises(ValueError, match="test_size"):
        split_train_test(X, y, test_size=1.2)

    with pytest.raises(ValueError, match="non-empty"):
        split_train_test(pd.DataFrame(), pd.Series(dtype=float))

    with pytest.raises(ValueError, match="does not match"):
        split_train_test(X, y.iloc[:-1])


def test_preprocessor_imputes_missing_and_avoids_leakage(
    sample_housing_df: pd.DataFrame,
) -> None:
    X, y = split_features_and_target(sample_housing_df)
    X_train, X_test, _, _ = split_train_test(X, y, test_size=0.33, random_state=42)

    X_train_missing = X_train.copy()
    X_test_extreme = X_test.copy()
    X_train_missing.iloc[0, 0] = np.nan
    X_test_extreme.iloc[0, 0] = 9999.0  # Must NOT influence train imputer/scaler

    preprocessor = build_preprocessor(
        numerical_features=list(X.columns),
        imputer_strategy="median",
        scale_features=True,
    )
    transformed_train = preprocessor.fit_transform(X_train_missing)
    transformed_test = preprocessor.transform(X_test_extreme)

    assert not np.isnan(transformed_train).any()
    assert transformed_train.shape == X_train_missing.shape
    assert transformed_test.shape == X_test_extreme.shape
    # Training mean after StandardScaler should be ~0 regardless of extreme test value
    assert np.allclose(transformed_train.mean(axis=0), 0.0, atol=1e-7)


def test_build_preprocessor_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_preprocessor(numerical_features=[])

    with pytest.raises(ValueError, match="Unsupported imputer_strategy"):
        build_preprocessor(numerical_features=["MedInc"], imputer_strategy="invalid")

