"""
Dependencies and Model Lifecycle Manager for FastAPI
Loads the production model from MLflow Model Registry or local fallback artifact.
"""

import os
import sys
from typing import Any

import joblib
import pandas as pd

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class ModelManager:
    """Singleton model lifecycle and inference manager."""

    def __init__(self):
        self.model: Any = None
        self.model_name: str = os.getenv("MODEL_NAME", "house-price-predictor")
        self.model_version: str = "unknown"
        self.source: str = "none"

    def load_model(self) -> None:
        """
        Attempt loading from MLflow Registry first; fall back to local disk serialized pipeline.
        """
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        loaded_from_mlflow = False

        if mlflow_uri:
            try:
                import urllib.request

                # Fast reachability probe so unreachable MLflow fails immediately instead of retrying for minutes
                if mlflow_uri.startswith("http"):
                    urllib.request.urlopen(mlflow_uri, timeout=2.0).close()

                os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
                os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

                import mlflow
                import mlflow.sklearn
                from mlflow.tracking import MlflowClient

                mlflow.set_tracking_uri(mlflow_uri)
                client = MlflowClient(tracking_uri=mlflow_uri)

                # Resolve available version URI cleanly via registry check before downloading
                candidate_uris = []
                try:
                    mv = client.get_model_version_by_alias(self.model_name, "champion")
                    candidate_uris.append((f"models:/{self.model_name}@champion", str(mv.version)))
                except Exception:
                    pass

                for fallback_stage in ("Production", "1"):
                    candidate_uris.append(
                        (f"models:/{self.model_name}/{fallback_stage}", fallback_stage)
                    )

                for target_uri, ver_label in candidate_uris:
                    try:
                        print(
                            f"[ModelManager] Attempting to load model from MLflow: {target_uri}..."
                        )
                        self.model = mlflow.sklearn.load_model(target_uri)
                        self.model_version = ver_label
                        self.source = f"mlflow ({target_uri})"
                        loaded_from_mlflow = True
                        print(f"[ModelManager] Successfully loaded from MLflow: {target_uri}")
                        break
                    except Exception:
                        continue

            except Exception as e:
                print(
                    f"[ModelManager] MLflow connection failed: {e}. Falling back to local artifact."
                )

        if not loaded_from_mlflow:
            # Fall back to local artifact
            local_pipeline_path = os.getenv("MODEL_LOCAL_PATH", "models/full_pipeline.joblib")
            if os.path.exists(local_pipeline_path):
                print(
                    f"[ModelManager] Loading local serialized pipeline from '{local_pipeline_path}'..."
                )
                self.model = joblib.load(local_pipeline_path)
                self.model_version = "1.0.0-local"
                self.source = f"local file ({local_pipeline_path})"
                print(f"[ModelManager] Successfully loaded model from '{local_pipeline_path}'")
            else:
                print(f"[ModelManager] WARNING: No local model found at '{local_pipeline_path}'!")
                self.model = None

    def predict(self, df_features: pd.DataFrame) -> list:
        """Score input feature dataframe."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot execute prediction.")
        predictions = self.model.predict(df_features)
        return [float(p) for p in predictions]


# Global singleton instance
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """FastAPI dependency to inject model manager."""
    return model_manager
