"""
Model Registry & Promotion Module for MLOps Lab
Handles registering models to MLflow Model Registry and promoting them
to Production/Champion status.
"""

import os
import warnings

import mlflow
from mlflow.tracking import MlflowClient


def register_and_promote_model(
    run_id: str,
    artifact_path: str,
    model_name: str,
    alias: str = "champion",
    stage: str = "Production",
    description: str | None = None,
    model_uri: str | None = None,
) -> str | None:
    """
    Register a model from an MLflow run and promote it to Production/Champion.

    Returns the new model version number as string, or None if failed.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    target_uri = model_uri or f"runs:/{run_id}/{artifact_path}"
    print(f"[Registry] Registering model from URI: {target_uri} into '{model_name}'...")

    try:
        # Create registered model if it does not exist
        try:
            client.get_registered_model(model_name)
        except Exception:
            client.create_registered_model(
                name=model_name,
                description=description or "Production house price regression model",
            )

        # Register model using mlflow.register_model (supports MLflow 3.x models:/m-... and runs:/ URIs)
        mv = mlflow.register_model(model_uri=target_uri, name=model_name)
        version = str(mv.version)
        print(f"[Registry] Created Model Version: {version} for '{model_name}'")

        # Set alias (modern MLflow standard)
        try:
            client.set_registered_model_alias(name=model_name, alias=alias, version=version)
            print(f"[Registry] Set alias '@{alias}' -> version {version}")
        except Exception as alias_err:
            print(f"[Registry] Note: Could not set alias: {alias_err}")

        # Set stage (backward-compatible MLflow transition when supported)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                client.transition_model_version_stage(
                    name=model_name, version=version, stage=stage, archive_existing_versions=False
                )
            print(f"[Registry] Transitioned version {version} to stage '{stage}'")
        except Exception:
            print(
                f"[Registry] Note: Legacy stage transition skipped (using MLflow alias '@{alias}')"
            )

        return version

    except Exception as e:
        print(f"[Registry] Warning: Could not complete remote registration: {e}")
        return None
