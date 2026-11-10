"""Model training pipeline, evaluation, selection, and artifact persistence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.data_loader import load_housing_dataset, split_features_and_target
from src.evaluate import (
    evaluate_models,
    explain_best_model,
    format_comparison_table,
    select_best_model,
)
from src.preprocessing import build_preprocessor, split_train_test
from src.utils import ensure_dir, load_config, save_json, setup_logging

logger = setup_logging()


def build_candidate_estimators(models_config: dict[str, Any]) -> dict[str, RegressorMixin]:
    """Instantiate candidate regression estimators from configuration.

    Args:
        models_config: 'models' section of config.yaml.

    Returns:
        Mapping from model identifier to Scikit-learn regressor instance.

    Raises:
        ValueError: If fewer than three models or invalid model keys are configured.
    """
    registry: dict[str, type[RegressorMixin]] = {
        "linear_regression": LinearRegression,
        "decision_tree": DecisionTreeRegressor,
        "random_forest": RandomForestRegressor,
    }

    estimators: dict[str, RegressorMixin] = {}
    for model_name, estimator_cls in registry.items():
        if model_name in models_config:
            params = models_config[model_name] or {}
            estimators[model_name] = estimator_cls(**params)

    if len(estimators) < 3:
        raise ValueError(
            "Expected at least 3 configured models (linear_regression, decision_tree, random_forest), "
            f"found: {list(estimators.keys())}."
        )
    return estimators


def train_candidate_pipelines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str],
    models_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
) -> dict[str, Pipeline]:
    """Build and fit one Scikit-learn Pipeline per candidate regression model."""
    estimators = build_candidate_estimators(models_config)
    imputer_strategy = str(preprocessing_config.get("imputer_strategy", "median"))
    scale_features = bool(preprocessing_config.get("scale_features", True))

    trained_pipelines: dict[str, Pipeline] = {}
    for model_name, estimator in estimators.items():
        preprocessor = build_preprocessor(
            numerical_features=feature_names,
            imputer_strategy=imputer_strategy,
            scale_features=scale_features,
        )
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        logger.info("Training model '%s' on %d samples...", model_name, len(X_train))
        pipeline.fit(X_train, y_train)
        trained_pipelines[model_name] = pipeline

    return trained_pipelines


def save_model_artifacts(
    best_model_name: str,
    best_pipeline: Pipeline,
    best_metrics: dict[str, float],
    results_df: pd.DataFrame,
    feature_names: list[str],
    target_name: str,
    config: dict[str, Any],
    train_samples: int,
    test_samples: int,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Save the best trained pipeline (model.pkl) and metadata JSON."""
    output_cfg = config["output"]
    target_dir = ensure_dir(output_dir if output_dir is not None else output_cfg["model_dir"])

    model_path = target_dir / output_cfg.get("model_filename", "model.pkl")
    metadata_path = target_dir / output_cfg.get("metadata_filename", "model_metadata.json")

    joblib.dump(best_pipeline, model_path)

    estimator = best_pipeline.named_steps.get("model")
    prep_cfg = config["preprocessing"]
    models_cfg = config["models"]
    eval_cfg = config["evaluation"]

    metadata: dict[str, Any] = {
        "best_model_name": best_model_name,
        "estimator_class": estimator.__class__.__name__ if estimator is not None else "Unknown",
        "dataset_name": config["dataset"].get("name", "california_housing"),
        "feature_names": feature_names,
        "target_name": target_name,
        "training_configuration": {
            "random_state": int(prep_cfg.get("random_state", 42)),
            "test_size": float(prep_cfg.get("test_size", 0.2)),
            "train_samples": train_samples,
            "test_samples": test_samples,
            "preprocessing": {
                "imputer_strategy": prep_cfg.get("imputer_strategy", "median"),
                "scale_features": bool(prep_cfg.get("scale_features", True)),
            },
            "model_hyperparameters": models_cfg.get(best_model_name, {}),
            "all_model_hyperparameters": models_cfg,
        },
        "evaluation_metrics": {
            "primary_metric": eval_cfg.get("primary_metric", "rmse"),
            "best_model_metrics": best_metrics,
            "all_models": results_df.to_dict(orient="records"),
        },
        "scikit_learn_version": sklearn.__version__,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    save_json(metadata, metadata_path)
    logger.info("Saved best model pipeline to %s", model_path)
    logger.info("Saved model metadata to %s", metadata_path)
    return model_path, metadata_path


def run_training_pipeline(
    config_path: str | Path = "config.yaml",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute end-to-end training, evaluation, model selection, and saving."""
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    prep_cfg = config["preprocessing"]
    models_cfg = config["models"]
    eval_cfg = config["evaluation"]

    df, detected_features, detected_target = load_housing_dataset(
        data_home=dataset_cfg.get("data_home")
    )
    target_col = dataset_cfg.get("target_column", detected_target)
    feature_names = dataset_cfg.get("feature_columns", detected_features)

    X, y = split_features_and_target(df, target_column=target_col)
    X = X[feature_names]

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=float(prep_cfg.get("test_size", 0.2)),
        random_state=int(prep_cfg.get("random_state", 42)),
    )
    logger.info(
        "Split dataset into %d training samples and %d test samples.",
        len(X_train),
        len(X_test),
    )

    trained_pipelines = train_candidate_pipelines(
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        models_config=models_cfg,
        preprocessing_config=prep_cfg,
    )

    results_df = evaluate_models(trained_pipelines, X_test, y_test)
    primary_metric = str(eval_cfg.get("primary_metric", "rmse"))
    best_model_name, best_metrics = select_best_model(results_df, primary_metric=primary_metric)

    print()
    print(format_comparison_table(results_df))
    print()
    print(explain_best_model(best_model_name, results_df))
    print()

    model_path, metadata_path = save_model_artifacts(
        best_model_name=best_model_name,
        best_pipeline=trained_pipelines[best_model_name],
        best_metrics=best_metrics,
        results_df=results_df,
        feature_names=feature_names,
        target_name=target_col,
        config=config,
        train_samples=len(X_train),
        test_samples=len(X_test),
        output_dir=output_dir,
    )

    return {
        "best_model_name": best_model_name,
        "best_metrics": best_metrics,
        "results": results_df,
        "model_path": model_path,
        "metadata_path": metadata_path,
    }


def main() -> None:
    """CLI entry point for training all models and saving the best one."""
    parser = argparse.ArgumentParser(description="Train regression models for house price prediction.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    args = parser.parse_args()
    run_training_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()

