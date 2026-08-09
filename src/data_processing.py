"""
Cleaning and feature engineering for the OLX Pakistan used-car listings.

Kept separate from the training script because I ended up reusing the same
cleaning logic in the exploration notebook and wanted one place to fix bugs
instead of two.
"""

import numpy as np
import pandas as pd

# "Range Rover" and "Land Rover" get listed as separate brands in the raw
# data, but Range Rover is a Land Rover model line, not its own
# manufacturer — same fix logic as the mislabeled brands, just a merge
# instead of a spelling correction. "Other Brands" is the source site's own
# catch-all category, so it gets folded into the "Other" bucket this script
# already builds for rare brands.
BRAND_FIXES = {
    "Range Rover": "Land Rover",
    "Other Brands": "Other",
}

# Columns a listing can't be usable without. Registered City is handled
# separately below (missing city doesn't make a listing unusable, it just
# means the seller didn't say where the car is).
REQUIRED_COLUMNS = ["Brand", "Condition", "Fuel", "KMs Driven", "Model",
                    "Price", "Transaction Type", "Year"]


def load_raw(path: str) -> pd.DataFrame:
    # The file has a couple of non-UTF-8 characters in city names
    # (Sheikhūpura and similar), so it needs latin-1 rather than the
    # pandas default.
    return pd.read_csv(path, encoding="latin1")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df.drop_duplicates()

    # This dataset mixes new and used listings. "Used Car Price Estimator"
    # means used — new-car listings are showroom prices, not resale value,
    # and mixing the two would teach the model the wrong relationship
    # between age and price.
    df = df[df["Condition"] == "Used"]

    df["Brand"] = df["Brand"].str.strip().replace(BRAND_FIXES)

    # Dataset's most recent listing year is 2019, so that's "now" for
    # computing car age — same reasoning as fixing depreciation to when the
    # data was actually collected rather than today's date.
    reference_year = int(df["Year"].max())
    df["car_age"] = reference_year - df["Year"]

    # Missing city isn't a broken row, it's a seller who didn't fill that
    # field in. Worth keeping as its own category rather than dropping the
    # listing or guessing a city.
    df["Registered City"] = df["Registered City"].fillna("Not Specified")

    # IQR-based outlier trim, widened to 3x IQR instead of the usual 1.5x
    # so real (if unusual) high-end cars and near-new low-mileage listings
    # don't get cut alongside actual data entry errors — and this dataset
    # has plenty of those: a handful of "cars" priced at multiple crore
    # rupees, one listing with a 1915 model year, and mileages past a
    # million km that no working car actually has.
    for col in ["Price", "KMs Driven", "car_age"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 3 * iqr, q3 + 3 * iqr
        df = df[(df[col] >= max(low, 0)) & (df[col] <= high)]

    # Brands and cities with only a handful of listings don't give the
    # model anything to learn and just add sparse one-hot columns.
    brand_counts = df["Brand"].value_counts()
    common_brands = brand_counts[brand_counts >= 5].index
    df["Brand"] = np.where(df["Brand"].isin(common_brands), df["Brand"], "Other")

    city_counts = df["Registered City"].value_counts()
    common_cities = city_counts[city_counts >= 10].index
    df["Registered City"] = np.where(
        df["Registered City"].isin(common_cities), df["Registered City"], "Other City"
    )

    df = df.rename(columns={
        "Brand": "brand",
        "Fuel": "fuel",
        "KMs Driven": "km_driven",
        "Registered City": "city",
        "Transaction Type": "transaction_type",
        "Price": "selling_price",
    })

    # Model (303 distinct values, most with a handful of listings) and
    # Condition (constant now that the dataset's filtered to "Used") don't
    # add anything a one-hot encoder could learn from, so they're dropped
    # here rather than carried through as dead weight.
    df = df.drop(columns=["Model", "Condition", "Year"])

    return df.reset_index(drop=True)


FEATURE_COLUMNS = ["brand", "car_age", "km_driven", "fuel", "city", "transaction_type"]
TARGET_COLUMN = "selling_price"
CATEGORICAL_COLUMNS = ["brand", "fuel", "city", "transaction_type"]
NUMERIC_COLUMNS = ["car_age", "km_driven"]


def load_and_clean(path: str) -> pd.DataFrame:
    return clean(load_raw(path))
