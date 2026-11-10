# House Price Prediction

A clean, modular, standalone Python Machine Learning application for predicting median California house values using Scikit-learn.

---

## 1. Project Overview

This project implements a supervised regression workflow from data loading and exploratory analysis to preprocessing, multi-model training, evaluation, artifact persistence, batch/single inference, and automated testing.

All configuration (split ratio, random seeds, preprocessing choices, model hyperparameters, and artifact paths) is centralized in `config.yaml`.

### Project Structure

```text
.
├── config.yaml
├── data/
│   └── README.md
├── models/
│   └── README.md
├── notebooks/
│   └── exploratory_analysis.ipynb
├── scripts/
│   └── predict_from_csv.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
├── tests/
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_prediction.py
│   ├── test_preprocessing.py
│   └── test_training.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 2. ML Problem Description

- **Task Type:** Supervised Machine Learning — Regression
- **Objective:** Predict the median house value (`MedHouseVal`, measured in units of \$100,000) for a California census block group given socioeconomic, housing, and geographic attributes.
- **Primary Evaluation Metric:** Root Mean Squared Error (RMSE) on the held-out test set (lower is better), accompanied by MAE, MSE, and R² Score.

---

## 3. Dataset Information

- **Source:** California Housing Dataset (`sklearn.datasets.fetch_california_housing`), derived from the 1990 U.S. Census.
- **Size:** 20,640 samples, 8 numerical input features, 1 continuous target variable.
- **Data Quality:** 0 missing values and 0 duplicate records in the standard dataset.
- **Train / Test Split:** 80% training (16,512 samples) / 20% testing (4,128 samples) with `random_state: 42`.

---

## 4. Feature Descriptions

| Feature | Description |
| --- | --- |
| `MedInc` | Median income in block group (in tens of thousands of USD) |
| `HouseAge` | Median house age in block group (years) |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |
| `MedHouseVal` *(target)* | Median house value for California districts (in \$100,000s) |

---

## 5. Model Descriptions

Every candidate model is wrapped inside a Scikit-learn `Pipeline` containing an unfitted `ColumnTransformer` (`SimpleImputer` with median strategy + `StandardScaler`) fitted strictly on training data to avoid data leakage:

1. **Linear Regression (`LinearRegression`)** — Baseline parametric model fitting an additive linear hyperplane across standardized features.
2. **Decision Tree Regressor (`DecisionTreeRegressor`)** — Non-linear tree model capturing feature thresholds and interactions (`max_depth=10`, `min_samples_split=5`, `min_samples_leaf=2`).
3. **Random Forest Regressor (`RandomForestRegressor`)** — Bootstrap-aggregated ensemble of decision trees (`n_estimators=100`, `max_depth=15`, `min_samples_split=5`, `min_samples_leaf=2`) that lowers prediction variance and captures complex spatial/income interactions.

---

## 6. Installation Instructions

Run from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 7. Data Loading & Exploratory Analysis

Load the California Housing dataset, inspect shape/columns, check missing values and duplicates, and print statistical summaries:

```bash
python -m src.data_loader
```

To also export a CSV copy to `data/california_housing.csv`:

```bash
python -m src.data_loader --export-csv
```

---

## 8. How to Run Training & Save the Best Model

Train all configured models, compare MAE/MSE/RMSE/R², select the lowest-RMSE model, and save `models/model.pkl` and `models/model_metadata.json`:

```bash
python -m src.train --config config.yaml
```

---

## 9. How to Evaluate Models

Re-evaluate the saved `models/model.pkl` on the configured test split and display the comparison table and selection explanation:

```bash
python -m src.evaluate --config config.yaml
```

---

## 10. How to Make Predictions

### Option A: Single or Batch JSON via Module CLI

```bash
python -m src.predict --input-json '{"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.9841, "AveBedrms": 1.0238, "Population": 322.0, "AveOccup": 2.5556, "Latitude": 37.88, "Longitude": -122.23}'
```

### Option B: Batch Prediction from a CSV File

```bash
python scripts/predict_from_csv.py --input path/to/input.csv --output path/to/predictions.csv
```

The input CSV must contain the 8 feature columns (`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`). Output includes `predicted_med_house_val` (in \$100,000s) and `predicted_price_usd` (in USD).

---

## 11. How to Run Tests

```bash
python -m pytest -v
```

---

## 12. Expected Output & Generated Files

### Observed Model Comparison Output

```text
====================================================================
MODEL EVALUATION COMPARISON (Sorted by Test RMSE)
====================================================================
            Model    MAE    MSE   RMSE     R2
    random_forest 0.3307 0.2587 0.5086 0.8026
    decision_tree 0.4303 0.4059 0.6371 0.6903
linear_regression 0.5332 0.5559 0.7456 0.5758
====================================================================
```

- **Selected Best Model:** `random_forest` (Test RMSE = `0.5086`, R² = `0.8026`), outperforming `decision_tree` by `0.1285` RMSE (20.2% lower error) and `linear_regression` by `0.2370` RMSE (31.8% lower error).

### Generated Files After Training

- `models/model.pkl` — Serialized Scikit-learn `Pipeline` containing both the fitted preprocessing transformer and the best trained regressor (`RandomForestRegressor`).
- `models/model_metadata.json` — JSON file recording `best_model_name`, `feature_names`, `training_configuration`, `evaluation_metrics` (best model + all candidate models), `scikit_learn_version`, and UTC timestamp.

