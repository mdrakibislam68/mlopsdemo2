"""Preprocessing utilities and train/test splitting without data leakage."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

VALID_IMPUTER_STRATEGIES = {"mean", "median", "most_frequent", "constant"}


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into reproducible train and test sets.

    Splitting happens prior to fitting any preprocessing transformers to
    prevent data leakage from the test set into training statistics.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Fraction of samples allocated to the test set (0 < test_size < 1).
        random_state: Seed for reproducible splitting.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If inputs are empty, mismatched in length, or test_size is invalid.
    """
    if X.empty or y.empty:
        raise ValueError("Input features and target must be non-empty.")
    if len(X) != len(y):
        raise ValueError(f"Feature row count ({len(X)}) does not match target length ({len(y)}).")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1 (exclusive), got {test_size}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(
    numerical_features: list[str],
    imputer_strategy: str = "median",
    scale_features: bool = True,
) -> ColumnTransformer:
    """Build an unfitted Scikit-learn preprocessing pipeline for numerical features.

    Args:
        numerical_features: List of numerical feature column names.
        imputer_strategy: Strategy for SimpleImputer ('mean', 'median', 'most_frequent', 'constant').
        scale_features: Whether to apply StandardScaler after imputation.

    Returns:
        Unfitted ColumnTransformer ready to be placed inside a model Pipeline.

    Raises:
        ValueError: If numerical_features is empty or imputer_strategy is unsupported.
    """
    if not numerical_features:
        raise ValueError("numerical_features must contain at least one feature name.")
    if imputer_strategy not in VALID_IMPUTER_STRATEGIES:
        raise ValueError(
            f"Unsupported imputer_strategy '{imputer_strategy}'. "
            f"Choose from {sorted(VALID_IMPUTER_STRATEGIES)}."
        )

    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
    ]
    if scale_features:
        steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numerical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor

