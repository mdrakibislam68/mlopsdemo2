"""
Data Preprocessing Module for MLOps Lab
Splits data into train/test sets, imputes missing values, encodes
categorical features, scales numerical features, and saves processed datasets.
"""

import argparse
import os

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer for numerical and categorical feature pipelines.
    - Numerical: median imputation + standard scaling
    - Categorical: most frequent imputation + one-hot encoding
    """
    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ]
    )
    # Ensure pandas DataFrame output with feature names preserved
    preprocessor.set_output(transform="pandas")
    return preprocessor


def preprocess_data(
    raw_data_path: str, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Load raw data, split into train and test, fit preprocessor on train,
    and return transformed DataFrames along with the fitted transformer.
    """
    df = pd.read_csv(raw_data_path)
    val_cfg = config["data_validation"]
    prep_cfg = config["preprocessing"]
    base_cfg = config["base"]

    target_col = val_cfg["target_column"]
    numerical_cols = val_cfg["numerical_columns"]
    categorical_cols = val_cfg["categorical_columns"]
    feature_cols = numerical_cols + categorical_cols

    # Ensure target column is dropped from features
    X = df[feature_cols]
    y = df[target_col]

    # Split train/test (prevents data leakage during transformation)
    test_size = prep_cfg.get("test_size", 0.2)
    random_state = base_cfg.get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Build and fit preprocessor on training data
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    preprocessor.fit(X_train)

    # Transform features using preprocessor (produces pandas DataFrame with imputed & scaled features)
    train_df = preprocessor.transform(X_train).copy()
    train_df[target_col] = y_train.values

    test_df = preprocessor.transform(X_test).copy()
    test_df[target_col] = y_test.values

    return train_df, test_df, preprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for model training")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    raw_path = config["data_generation"]["raw_data_path"]
    train_path = config["preprocessing"]["train_data_path"]
    test_path = config["preprocessing"]["test_data_path"]
    preprocessor_path = config["preprocessing"]["preprocessor_path"]

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    print(f"[Preprocess] Reading from '{raw_path}'...")
    train_df, test_df, preprocessor = preprocess_data(raw_path, config)

    # Save processed CSVs and preprocessor artifact
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    joblib.dump(preprocessor, preprocessor_path)

    print(f"[Preprocess] Saved train dataset ({train_df.shape}) to '{train_path}'")
    print(f"[Preprocess] Saved test dataset  ({test_df.shape}) to '{test_path}'")
    print(f"[Preprocess] Serialized preprocessor to '{preprocessor_path}'")


if __name__ == "__main__":
    main()
