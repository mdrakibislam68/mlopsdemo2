"""
Data Generation Module for MLOps Lab
Generates a realistic, reproducible house price dataset with continuous,
discrete, and categorical features with non-linear and linear relationships.
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml


def generate_housing_data(
    num_samples: int = 2000, random_state: int = 42, noise_factor: float = 15000.0
) -> pd.DataFrame:
    """
    Generate synthetic house price dataset with realistic relationships.

    Formula logic:
    base_price = 50,000
    + area * 140 (per sqft)
    + bedrooms * 18,000
    + bathrooms * 22,000
    + garage * 15,000
    - age * 1,200
    * location_multiplier (urban: 1.35, suburban: 1.10, rural: 0.85)
    + Gaussian noise
    """
    np.random.seed(random_state)

    # 1. Continuous feature: Area (sq ft) - log-normal distribution shifted to ~800 - 4500
    area = np.random.normal(loc=2100, scale=650, size=num_samples).astype(int)
    area = np.clip(area, 500, 5500)

    # 2. Discrete features: Bedrooms, Bathrooms, Garage, Age
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=num_samples, p=[0.05, 0.25, 0.45, 0.20, 0.05])
    bathrooms = np.round(bedrooms * np.random.uniform(0.6, 1.1, size=num_samples) * 2) / 2
    bathrooms = np.clip(bathrooms, 1.0, 5.0)

    age = np.random.randint(0, 55, size=num_samples)
    garage = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.15, 0.35, 0.40, 0.10])

    # 3. Categorical feature: Location
    locations = np.random.choice(
        ["urban", "suburban", "rural"], size=num_samples, p=[0.40, 0.45, 0.15]
    )
    loc_multipliers = {"urban": 1.35, "suburban": 1.10, "rural": 0.85}
    loc_mult_values = np.array([loc_multipliers[loc] for loc in locations])

    # 4. Target variable: Price
    base = 60000.0
    area_contrib = area * 145.0
    bed_contrib = bedrooms * 16000.0
    bath_contrib = bathrooms * 24000.0
    garage_contrib = garage * 18000.0
    age_penalty = age * 1100.0

    raw_price = (
        base + area_contrib + bed_contrib + bath_contrib + garage_contrib - age_penalty
    ) * loc_mult_values
    noise = np.random.normal(loc=0, scale=noise_factor, size=num_samples)
    price = np.round(raw_price + noise, -2)  # round to nearest $100
    price = np.clip(price, 45000, 2500000)

    df = pd.DataFrame(
        {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "location": locations,
            "age": age,
            "garage": garage,
            "price": price,
        }
    )

    # Introduce a tiny realistic percentage of missing values (0.5%) to test imputation handling
    mask_bath = np.random.rand(num_samples) < 0.005
    df.loc[mask_bath, "bathrooms"] = np.nan

    mask_garage = np.random.rand(num_samples) < 0.005
    df.loc[mask_garage, "garage"] = np.nan

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic house price dataset")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)

    num_samples = config["data_generation"]["num_samples"]
    noise_factor = config["data_generation"]["noise_factor"]
    random_state = config["base"]["random_state"]
    output_path = args.output or config["data_generation"]["raw_data_path"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[DataGen] Generating {num_samples} samples with random_state={random_state}...")
    df = generate_housing_data(
        num_samples=num_samples, random_state=random_state, noise_factor=noise_factor
    )
    df.to_csv(output_path, index=False)
    print(
        f"[DataGen] Successfully saved raw dataset to '{output_path}' ({len(df)} rows, {len(df.columns)} columns)"
    )
    print(f"[DataGen] Preview:\n{df.head(3)}")


if __name__ == "__main__":
    main()
