"""
Unit tests for data generation, validation, and preprocessing modules.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.generate_data import generate_housing_data
from src.data.preprocess import build_preprocessor
from src.data.validate_data import validate_schema


@pytest.fixture
def sample_config():
    return {
        "base": {"random_state": 42},
        "data_validation": {
            "required_columns": [
                "area",
                "bedrooms",
                "bathrooms",
                "location",
                "age",
                "garage",
                "price",
            ],
            "numerical_columns": ["area", "bedrooms", "bathrooms", "age", "garage"],
            "categorical_columns": ["location"],
            "target_column": "price",
            "ranges": {
                "area": [300, 10000],
                "bedrooms": [1, 10],
                "bathrooms": [1.0, 8.0],
                "age": [0, 100],
                "garage": [0, 5],
                "price": [30000, 3000000],
            },
        },
        "preprocessing": {"test_size": 0.2},
    }


def test_data_generation_shape_and_columns():
    """Verify data generator creates expected rows, columns, and non-empty dataframe."""
    df = generate_housing_data(num_samples=100, random_state=42)
    assert len(df) == 100
    expected_cols = {"area", "bedrooms", "bathrooms", "location", "age", "garage", "price"}
    assert set(df.columns) == expected_cols
    assert (df["area"] > 0).all()
    assert (df["price"] > 0).all()


def test_data_validation_success(sample_config):
    """Verify valid dataset passes validation."""
    df = generate_housing_data(num_samples=50, random_state=42)
    # Fill any injected NaNs for strict test
    df["bathrooms"] = df["bathrooms"].fillna(2.0)
    df["garage"] = df["garage"].fillna(1.0)

    is_valid, errors = validate_schema(df, sample_config)
    assert is_valid is True
    assert len(errors) == 0


def test_data_validation_missing_column(sample_config):
    """Verify validation detects missing required columns."""
    df = generate_housing_data(num_samples=50, random_state=42)
    df = df.drop(columns=["price"])

    is_valid, errors = validate_schema(df, sample_config)
    assert is_valid is False
    assert any("Missing required columns" in e for e in errors)


def test_data_validation_invalid_category(sample_config):
    """Verify validation detects invalid categorical values."""
    df = generate_housing_data(num_samples=50, random_state=42)
    df.loc[0, "location"] = "mars_colony"

    is_valid, errors = validate_schema(df, sample_config)
    assert is_valid is False
    assert any("unexpected categories" in e for e in errors)


def test_data_validation_out_of_range(sample_config):
    """Verify validation detects values outside allowed ranges."""
    df = generate_housing_data(num_samples=50, random_state=42)
    df.loc[0, "area"] = 99999  # Above 10,000 max

    is_valid, errors = validate_schema(df, sample_config)
    assert is_valid is False
    assert any("above maximum" in e for e in errors)


def test_build_preprocessor():
    """Verify preprocessor handles numeric and categorical columns."""
    num_cols = ["area", "bedrooms"]
    cat_cols = ["location"]

    prep = build_preprocessor(num_cols, cat_cols)
    sample_df = pd.DataFrame(
        {
            "area": [1500, 2000, np.nan],
            "bedrooms": [3, 4, 2],
            "location": ["urban", "suburban", "rural"],
        }
    )

    prep.fit(sample_df)
    transformed = prep.transform(sample_df)

    # 2 numericals + 3 one-hot categories (urban, suburban, rural) = 5 columns
    assert transformed.shape == (3, 5)
    assert not transformed.isna().any().any()
