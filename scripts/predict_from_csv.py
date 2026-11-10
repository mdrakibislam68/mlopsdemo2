"""CLI script for batch house price prediction from a CSV file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly as `python scripts/predict_from_csv.py` from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH, predict_from_csv


def main() -> None:
    """Parse arguments and run CSV batch prediction."""
    parser = argparse.ArgumentParser(
        description="Run batch house price predictions from a CSV file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file containing housing features.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path to write predictions.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to trained model.pkl file.",
    )
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA_PATH),
        help="Path to model_metadata.json file.",
    )
    args = parser.parse_args()

    results = predict_from_csv(
        input_csv=args.input,
        output_csv=args.output,
        model_path=args.model,
        metadata_path=args.metadata,
    )
    print("\nHouse Price Predictions:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

