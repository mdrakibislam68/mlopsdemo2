# Data Directory

This directory stores local dataset exports or sample input CSV files used for batch inference.

The primary dataset is the **California Housing Dataset**, loaded programmatically via `sklearn.datasets.fetch_california_housing`.

Running `python -m src.data_loader --export-csv` will optionally export the dataset to `data/california_housing.csv` for inspection.

