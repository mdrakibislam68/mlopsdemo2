"""
Model Training and Experiment Tracking Module for MLOps Lab
Trains Linear Regression and Random Forest models, logs parameters and metrics
to MLflow, compares models on test RMSE, and registers the winning champion model.
"""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import json
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.evaluation.evaluate import compute_regression_metrics
from src.training.registry import register_and_promote_model

# Silence informational agent hints
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


def setup_mlflow(experiment_name: str) -> None:
    """Configure MLflow tracking URI and experiment with responsive fallback."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    # If an HTTP URL is specified, test connectivity with a fast timeout
    if tracking_uri.startswith("http"):
        import urllib.request

        try:
            req = urllib.request.Request(tracking_uri, method="GET")
            urllib.request.urlopen(req, timeout=1.5)
        except Exception:
            print(
                f"[MLflow] Server at {tracking_uri} not reachable; falling back to local SQLite 'sqlite:///mlflow.db'."
            )
            tracking_uri = "sqlite:///mlflow.db"

    mlflow.set_tracking_uri(tracking_uri)
    print(f"[MLflow] Active Tracking URI: {tracking_uri}")

    try:
        mlflow.set_experiment(experiment_name)
        print(f"[MLflow] Active Experiment: '{experiment_name}'")
    except Exception as e:
        print(f"[MLflow] Note: Experiment setup: {e}")


def train_and_log_model(
    model_name: str,
    model: Any,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: Any,
    raw_sample: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Fit a model, compute metrics, log full inference pipeline to MLflow, and return results.
    """
    print(f"\n[Training] --- Training {model_name} ---")
    model.fit(X_train, y_train)

    # Evaluate on both train (to check overfitting) and test (for model selection)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_metrics = compute_regression_metrics(y_train.values, train_preds)
    test_metrics = compute_regression_metrics(y_test.values, test_preds)

    print(
        f"[{model_name}] Train Metrics: RMSE={train_metrics['rmse']:,.2f}, R2={train_metrics['r2_score']:.4f}"
    )
    print(
        f"[{model_name}] Test  Metrics: RMSE={test_metrics['rmse']:,.2f}, R2={test_metrics['r2_score']:.4f}, MAE={test_metrics['mae']:,.2f}"
    )

    # Wrap preprocessor + fitted estimator so registered model accepts raw API features directly
    inference_pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", model)])

    run_id = None
    model_uri = None
    try:
        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id
            print(f"[{model_name}] MLflow Run ID: {run_id}")

            # Log tags and params
            mlflow.set_tag("model_family", model_name)
            mlflow.log_param("model_type", model_name)
            for k, v in params.items():
                mlflow.log_param(k, v)

            # Log train & test metrics
            for k, v in train_metrics.items():
                mlflow.log_metric(f"train_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            # Infer signature on raw schema when available, otherwise processed schema
            if raw_sample is not None:
                model_to_log = inference_pipeline
                input_sample = raw_sample
            else:
                model_to_log = model
                input_sample = X_test.head(5)

            output_sample = model_to_log.predict(input_sample)
            signature = infer_signature(input_sample, output_sample)

            # Log model artifact with cloudpickle serialization for maximum compatibility
            model_info = mlflow.sklearn.log_model(
                sk_model=model_to_log,
                name="model",
                signature=signature,
                input_example=input_sample.iloc[[0]],
                serialization_format="cloudpickle",
            )
            model_uri = model_info.model_uri

    except Exception as e:
        print(
            f"[{model_name}] MLflow logging warning: {e}. (Proceeding with local artifact persistence)"
        )

    return {
        "model_name": model_name,
        "model_instance": model,
        "params": params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "run_id": run_id,
        "model_uri": model_uri,
    }


def main():
    parser = argparse.ArgumentParser(description="Train house price regression models")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_path = config["preprocessing"]["train_data_path"]
    test_path = config["preprocessing"]["test_data_path"]
    preprocessor_path = config["preprocessing"]["preprocessor_path"]
    target_col = config["data_validation"]["target_column"]
    exp_name = config["training"]["experiment_name"]
    reg_model_name = config["training"]["registered_model_name"]
    primary_metric = config["training"].get("primary_metric", "rmse")

    # Connect to MLflow
    setup_mlflow(exp_name)

    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    preprocessor = joblib.load(preprocessor_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    raw_sample = None
    raw_data_path = config.get("data_generation", {}).get("raw_data_path")
    if raw_data_path and os.path.exists(raw_data_path):
        raw_df = pd.read_csv(raw_data_path)
        raw_sample = raw_df.drop(columns=[target_col], errors="ignore").head(5)

    # Model 1: Linear Regression
    lr_params = config["training"]["models"]["linear_regression"]
    lr_model = LinearRegression(fit_intercept=lr_params.get("fit_intercept", True))
    lr_results = train_and_log_model(
        model_name="LinearRegression",
        model=lr_model,
        params=lr_params,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        preprocessor=preprocessor,
        raw_sample=raw_sample,
    )

    # Model 2: Random Forest Regressor
    rf_params = config["training"]["models"]["random_forest"]
    rf_model = RandomForestRegressor(
        n_estimators=rf_params.get("n_estimators", 100),
        max_depth=rf_params.get("max_depth", 12),
        min_samples_split=rf_params.get("min_samples_split", 4),
        min_samples_leaf=rf_params.get("min_samples_leaf", 2),
        random_state=rf_params.get("random_state", 42),
    )
    rf_results = train_and_log_model(
        model_name="RandomForestRegressor",
        model=rf_model,
        params=rf_params,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        preprocessor=preprocessor,
        raw_sample=raw_sample,
    )

    all_results = [lr_results, rf_results]

    # Automatic selection based on test RMSE (lower is better)
    best_result = min(all_results, key=lambda r: r["test_metrics"][primary_metric])
    best_name = best_result["model_name"]
    best_model = best_result["model_instance"]
    best_rmse = best_result["test_metrics"]["rmse"]
    best_r2 = best_result["test_metrics"]["r2_score"]

    print("\n==========================================")
    print(f"[Model Selection] Winning Model: '{best_name}'")
    print(
        f"Selection Metric: Test {primary_metric.upper()} = {best_rmse:,.2f} (R² = {best_r2:.4f})"
    )
    print("==========================================\n")

    # Register best model in MLflow Registry if run_id exists
    model_version = None
    if best_result["run_id"]:
        model_version = register_and_promote_model(
            run_id=best_result["run_id"],
            artifact_path="model",
            model_name=reg_model_name,
            alias="champion",
            stage="Production",
            model_uri=best_result.get("model_uri"),
        )

    # Persist local copy in models/ directory for decoupled serving & testing
    os.makedirs("models", exist_ok=True)
    best_model_path = "models/best_model.joblib"
    best_prep_path = "models/preprocessor.joblib"
    full_pipe_path = "models/full_pipeline.joblib"

    joblib.dump(best_model, best_model_path)
    joblib.dump(preprocessor, best_prep_path)

    # Build end-to-end inference pipeline: Raw Features -> Preprocessor -> Regressor
    full_pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", best_model)])
    joblib.dump(full_pipeline, full_pipe_path)

    print(f"[Artifacts] Persisted '{best_model_path}', '{best_prep_path}', and '{full_pipe_path}'")

    # Save summary metrics report
    report = {
        "best_model_name": best_name,
        "selection_metric": primary_metric,
        "registered_model_name": reg_model_name,
        "registered_version": model_version,
        "models_comparison": {
            r["model_name"]: {
                "test_metrics": r["test_metrics"],
                "train_metrics": r["train_metrics"],
                "run_id": r["run_id"],
            }
            for r in all_results
        },
    }

    with open("evaluation_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[Report] Saved evaluation comparison to 'evaluation_metrics.json'")


if __name__ == "__main__":
    main()
