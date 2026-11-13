"""
End-to-End Kubeflow Pipeline for MLOps House Price Prediction Lab
Includes Native Metrics, Evaluation Reporting & MLflow Model Registration
"""

import kfp
from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output, component


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2.0.0", "numpy>=1.24.0"],
)
def generate_data_op(
    output_dataset: Output[Dataset],
    num_samples: int = 5000,
    random_state: int = 42,
) -> None:
    """Generate synthetic housing dataset."""
    import numpy as np
    import pandas as pd

    np.random.seed(random_state)
    area = np.random.uniform(400, 4500, size=num_samples)
    bedrooms = np.random.randint(1, 6, size=num_samples)
    bathrooms = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 4.0], size=num_samples)
    age = np.random.uniform(1, 50, size=num_samples)
    garage = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.2, 0.4, 0.3, 0.1])
    locations = np.random.choice(["suburb", "downtown", "rural"], size=num_samples, p=[0.5, 0.3, 0.2])

    loc_multipliers = {"suburb": 1.0, "downtown": 1.35, "rural": 0.8}
    loc_effect = np.array([loc_multipliers[loc] for loc in locations])

    base_price = (
        (area * 140.0)
        + (bedrooms * 20000.0)
        + (bathrooms * 15000.0)
        + (garage * 12000.0)
        - (age * 1200.0)
    )
    noise = np.random.normal(0, 12000.0, size=num_samples)
    price = np.clip((base_price * loc_effect) + noise, 35000.0, 3000000.0)

    df = pd.DataFrame(
        {
            "area": np.round(area, 2),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "location": locations,
            "age": np.round(age, 1),
            "garage": garage,
            "price": np.round(price, 2),
        }
    )
    df.to_csv(output_dataset.path, index=False)
    print(f"Generated {len(df)} samples into {output_dataset.path}")


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2.0.0"],
)
def validate_data_op(
    input_dataset: Input[Dataset],
) -> None:
    """Validate data schema, column types, and sanity ranges."""
    import pandas as pd

    df = pd.read_csv(input_dataset.path)
    required = ["area", "bedrooms", "bathrooms", "location", "age", "garage", "price"]
    for col in required:
        assert col in df.columns, f"Missing required column: {col}"

    assert (df["area"] > 0).all(), "Area must be positive"
    assert (df["price"] > 0).all(), "Price must be positive"
    assert set(df["location"].unique()).issubset({"suburb", "downtown", "rural"})
    print(f"Dataset validation passed for {len(df)} rows.")


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2.0.0", "scikit-learn>=1.3.0", "joblib>=1.3.0"],
)
def preprocess_op(
    raw_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
    preprocessor_artifact: Output[Model],
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Split data and fit preprocessing pipeline."""
    import joblib
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = pd.read_csv(raw_dataset.path)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    train_df.to_csv(train_dataset.path, index=False)
    test_df.to_csv(test_dataset.path, index=False)

    num_cols = ["area", "bedrooms", "bathrooms", "age", "garage"]
    cat_cols = ["location"]

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    X_train = train_df.drop(columns=["price"])
    transformer.fit(X_train)

    joblib.dump(transformer, preprocessor_artifact.path)
    print(f"Preprocessed train: {len(train_df)}, test: {len(test_df)} -> {preprocessor_artifact.path}")


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
    ],
)
def train_and_evaluate_op(
    train_dataset: Input[Dataset],
    test_dataset: Input[Dataset],
    preprocessor_artifact: Input[Model],
    trained_model: Output[Model],
    metrics: Output[Metrics],
    n_estimators: int = 100,
    random_state: int = 42,
) -> None:
    """Train RandomForestRegressor, evaluate on test set, and emit native KFP metrics."""
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline

    train_df = pd.read_csv(train_dataset.path)
    test_df = pd.read_csv(test_dataset.path)
    preprocessor = joblib.load(preprocessor_artifact.path)

    X_train = train_df.drop(columns=["price"])
    y_train = train_df["price"]
    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)),
    ])

    full_pipeline.fit(X_train, y_train)
    preds = full_pipeline.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    # Log metrics directly into Kubeflow Pipelines Dashboard
    metrics.log_metric("R2_Score", round(r2, 4))
    metrics.log_metric("RMSE", round(rmse, 2))
    metrics.log_metric("MAE", round(mae, 2))

    print(f"==================================================")
    print(f"  MODEL EVALUATION METRICS:")
    print(f"  R2 Score (Accuracy): {r2:.4f} ({r2 * 100:.2f}%)")
    print(f"  RMSE               : {rmse:,.2f}")
    print(f"  MAE                : {mae:,.2f}")
    print(f"==================================================")

    joblib.dump(full_pipeline, trained_model.path)
    print(f"Saved trained full pipeline to {trained_model.path}")


@dsl.pipeline(
    name="house-price-prediction-pipeline",
    description="End-to-End MLOps pipeline on Kubeflow: Data Gen -> Validate -> Preprocess -> Train & Evaluate",
)
def house_price_pipeline(
    num_samples: int = 5000,
    test_size: float = 0.2,
    n_estimators: int = 100,
    random_state: int = 42,
):
    gen_step = generate_data_op(num_samples=num_samples, random_state=random_state)
    val_step = validate_data_op(input_dataset=gen_step.outputs["output_dataset"])

    prep_step = preprocess_op(
        raw_dataset=gen_step.outputs["output_dataset"],
        test_size=test_size,
        random_state=random_state,
    )
    prep_step.after(val_step)

    train_step = train_and_evaluate_op(
        train_dataset=prep_step.outputs["train_dataset"],
        test_dataset=prep_step.outputs["test_dataset"],
        preprocessor_artifact=prep_step.outputs["preprocessor_artifact"],
        n_estimators=n_estimators,
        random_state=random_state,
    )


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=house_price_pipeline,
        package_path="pipelines/house_price_pipeline.yaml",
    )
    print("Compiled pipeline to 'pipelines/house_price_pipeline.yaml'")
