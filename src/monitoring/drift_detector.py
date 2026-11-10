"""
Data Drift Detection Module for MLOps Lab
Uses statistical hypothesis testing (Kolmogorov-Smirnov two-sample test for numericals
and Chi-Square contingency test for categoricals) to detect distribution drift
between production inference traffic and baseline training data.
"""

import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class DriftDetector:
    """
    Statistical Data Drift Detector comparing production batches
    against baseline training data distributions.
    """

    def __init__(
        self, reference_data_path: str = "data/raw/housing.csv", p_value_threshold: float = 0.05
    ):
        self.reference_data_path = reference_data_path
        self.p_value_threshold = p_value_threshold
        self.reference_df: pd.DataFrame = None
        self._load_reference()

    def _load_reference(self) -> None:
        if os.path.exists(self.reference_data_path):
            self.reference_df = pd.read_csv(self.reference_data_path)
        else:
            raise FileNotFoundError(
                f"Reference baseline data not found at: {self.reference_data_path}"
            )

    def detect_numerical_drift(
        self, current_series: pd.Series, feature_name: str
    ) -> dict[str, Any]:
        """
        Run two-sample Kolmogorov-Smirnov test.
        H0: Both samples are drawn from the same continuous distribution.
        If p_value < threshold, reject H0 -> Drift detected!
        """
        ref_values = self.reference_df[feature_name].dropna().values
        curr_values = current_series.dropna().values

        if len(curr_values) < 5:
            return {
                "feature": feature_name,
                "type": "numerical",
                "test": "ks_2samp",
                "drift_detected": False,
                "p_value": 1.0,
                "statistic": 0.0,
                "note": "Sample size too small for statistical significance",
            }

        stat, p_val = stats.ks_2samp(ref_values, curr_values)
        drift = bool(p_val < self.p_value_threshold)

        return {
            "feature": feature_name,
            "type": "numerical",
            "test": "ks_2samp",
            "drift_detected": drift,
            "p_value": round(float(p_val), 5),
            "statistic": round(float(stat), 5),
            "baseline_mean": round(float(np.mean(ref_values)), 2),
            "current_mean": round(float(np.mean(curr_values)), 2),
        }

    def detect_categorical_drift(
        self, current_series: pd.Series, feature_name: str
    ) -> dict[str, Any]:
        """
        Run Chi-Square goodness-of-fit test comparing category proportions.
        """
        ref_counts = self.reference_df[feature_name].value_counts(normalize=True)
        curr_counts = current_series.value_counts()

        categories = ref_counts.index.tolist()
        observed = [curr_counts.get(cat, 0) for cat in categories]
        total_curr = sum(observed)

        if total_curr < 10:
            return {
                "feature": feature_name,
                "type": "categorical",
                "test": "chisquare",
                "drift_detected": False,
                "p_value": 1.0,
                "statistic": 0.0,
                "note": "Sample size too small for chi-square test",
            }

        expected = [ref_counts[cat] * total_curr for cat in categories]
        stat, p_val = stats.chisquare(f_obs=observed, f_exp=expected)
        drift = bool(p_val < self.p_value_threshold)

        return {
            "feature": feature_name,
            "type": "categorical",
            "test": "chisquare",
            "drift_detected": drift,
            "p_value": round(float(p_val), 5),
            "statistic": round(float(stat), 5),
            "baseline_distribution": ref_counts.to_dict(),
            "current_distribution": (curr_counts / total_curr).to_dict(),
        }

    def analyze_batch(self, current_df: pd.DataFrame) -> dict[str, Any]:
        """
        Perform drift detection across all matching features in the batch.
        """
        results: dict[str, Any] = {}
        drifted_features: list[str] = []

        numerical_features = ["area", "bedrooms", "bathrooms", "age", "garage"]
        categorical_features = ["location"]

        for col in numerical_features:
            if col in current_df.columns:
                res = self.detect_numerical_drift(current_df[col], col)
                results[col] = res
                if res["drift_detected"]:
                    drifted_features.append(col)

        for col in categorical_features:
            if col in current_df.columns:
                res = self.detect_categorical_drift(current_df[col], col)
                results[col] = res
                if res["drift_detected"]:
                    drifted_features.append(col)

        overall_drift = len(drifted_features) > 0

        return {
            "overall_drift_detected": overall_drift,
            "drifted_features_count": len(drifted_features),
            "drifted_features": drifted_features,
            "p_value_threshold": self.p_value_threshold,
            "batch_size": len(current_df),
            "feature_details": results,
        }
