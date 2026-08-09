"""
Cleaning and feature engineering for the CarDekho used-car listings.

Kept separate from the training script because I ended up reusing the same
cleaning logic in the exploration notebook and wanted one place to fix bugs
instead of two.
"""

import numpy as np
import pandas as pd

# A few names in the raw "name" column are mangled ("OpelCorsa", "Land" for
# "Land Rover"). Rather than write a rule for every brand, I just fix the
# ones that actually show up in this dataset.
BRAND_FIXES = {
    "Opelcorsa": "Opel",
    "Land": "Land Rover",
    "Bmw": "BMW",
    "Mg": "MG",
}

# Dataset was scraped in 2020 (last listing year in the file), so that's
# what "car age" is measured against rather than today's date.
REFERENCE_YEAR = 2020


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates()

    # The columns that matter for prediction can't be null in a real
    # listing (a car with no year or no price isn't a usable row), so drop
    # rather than impute those. km_driven getting imputed with the median
    # is defensible if it's ever missing.
    required = ["name", "year", "selling_price", "km_driven", "fuel",
                "seller_type", "transmission", "owner"]
    df = df.dropna(subset=[c for c in required if c != "km_driven"])
    if df["km_driven"].isnull().any():
        df["km_driven"] = df["km_driven"].fillna(df["km_driven"].median())

    df["brand"] = df["name"].str.split().str[0].str.strip().str.title()
    df["brand"] = df["brand"].replace(BRAND_FIXES)

    df["car_age"] = REFERENCE_YEAR - df["year"]

    # A handful of "Test Drive Car" listings are dealer demo units with
    # near-zero mileage and prices that don't follow normal depreciation.
    # They skew the model without adding anything useful, so they're out.
    df = df[df["owner"] != "Test Drive Car"]

    # IQR-based outlier trim on price and mileage. Widened to 3x IQR
    # instead of the usual 1.5x because legitimate luxury cars and
    # near-new low-mileage cars would otherwise get cut.
    for col in ["selling_price", "km_driven"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 3 * iqr, q3 + 3 * iqr
        df = df[(df[col] >= max(low, 0)) & (df[col] <= high)]

    # Brands with a handful of listings (Jeep, MG, Ambassador...) don't
    # give the model anything to learn and just add sparse one-hot columns.
    brand_counts = df["brand"].value_counts()
    common_brands = brand_counts[brand_counts >= 5].index
    df["brand"] = np.where(df["brand"].isin(common_brands), df["brand"], "Other")

    df = df.drop(columns=["name", "year"])

    return df.reset_index(drop=True)


FEATURE_COLUMNS = ["brand", "car_age", "km_driven", "fuel", "seller_type",
                    "transmission", "owner"]
TARGET_COLUMN = "selling_price"
CATEGORICAL_COLUMNS = ["brand", "fuel", "seller_type", "transmission", "owner"]
NUMERIC_COLUMNS = ["car_age", "km_driven"]


def load_and_clean(path: str) -> pd.DataFrame:
    return clean(load_raw(path))
