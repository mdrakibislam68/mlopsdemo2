"""
Unit tests for data drift detection module.
"""

import os

import pytest

from src.monitoring.drift_detector import DriftDetector


@pytest.fixture
def drift_detector():
    ref_path = "data/raw/housing.csv"
    if not os.path.exists(ref_path):
        pytest.skip("Reference dataset data/raw/housing.csv not present for test")
    return DriftDetector(reference_data_path=ref_path)


def test_drift_detector_initialization(drift_detector):
    """Verify reference data loads with required columns."""
    assert drift_detector.reference_df is not None
    assert len(drift_detector.reference_df) > 0
    assert "area" in drift_detector.reference_df.columns


def test_drift_detector_on_similar_distribution(drift_detector):
    """A sample drawn directly from the reference data should not detect extreme drift."""
    sample_df = drift_detector.reference_df.sample(n=100, random_state=42).copy()
    report = drift_detector.analyze_batch(sample_df)

    assert "overall_drift_detected" in report
    assert "feature_details" in report
    assert report["batch_size"] == 100
    # KS test p-value should be well above 0.05 for sample from same population
    assert report["feature_details"]["area"]["p_value"] > 0.05


def test_drift_detector_on_heavily_shifted_distribution(drift_detector):
    """Artificially shifted area and location must trigger drift detection."""
    sample_df = drift_detector.reference_df.sample(n=100, random_state=42).copy()
    # Heavily shift area to 9,000 sq ft (baseline mean is ~2,100)
    sample_df["area"] = 9000.0

    report = drift_detector.analyze_batch(sample_df)
    assert report["overall_drift_detected"] is True
    assert "area" in report["drifted_features"]
    assert report["feature_details"]["area"]["drift_detected"] is True
    assert report["feature_details"]["area"]["p_value"] < 0.01
