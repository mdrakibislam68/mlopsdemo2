"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_housing_df() -> pd.DataFrame:
    """Return a small synthetic housing DataFrame matching California Housing schema."""
    return pd.DataFrame(
        [
            {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.9841,
                "AveBedrms": 1.0238,
                "Population": 322.0,
                "AveOccup": 2.5556,
                "Latitude": 37.88,
                "Longitude": -122.23,
                "MedHouseVal": 4.526,
            },
            {
                "MedInc": 8.3014,
                "HouseAge": 21.0,
                "AveRooms": 6.2381,
                "AveBedrms": 0.9719,
                "Population": 2401.0,
                "AveOccup": 2.1098,
                "Latitude": 37.86,
                "Longitude": -122.22,
                "MedHouseVal": 3.585,
            },
            {
                "MedInc": 7.2574,
                "HouseAge": 52.0,
                "AveRooms": 8.2881,
                "AveBedrms": 1.0734,
                "Population": 496.0,
                "AveOccup": 2.8023,
                "Latitude": 37.85,
                "Longitude": -122.24,
                "MedHouseVal": 3.521,
            },
            {
                "MedInc": 5.6431,
                "HouseAge": 52.0,
                "AveRooms": 5.8174,
                "AveBedrms": 1.0731,
                "Population": 558.0,
                "AveOccup": 2.5479,
                "Latitude": 37.85,
                "Longitude": -122.25,
                "MedHouseVal": 3.413,
            },
            {
                "MedInc": 3.8462,
                "HouseAge": 52.0,
                "AveRooms": 6.2819,
                "AveBedrms": 1.0811,
                "Population": 565.0,
                "AveOccup": 2.1815,
                "Latitude": 37.85,
                "Longitude": -122.25,
                "MedHouseVal": 3.422,
            },
            {
                "MedInc": 4.0368,
                "HouseAge": 52.0,
                "AveRooms": 4.7617,
                "AveBedrms": 1.1036,
                "Population": 413.0,
                "AveOccup": 2.1399,
                "Latitude": 37.85,
                "Longitude": -122.25,
                "MedHouseVal": 2.697,
            },
        ]
    )

