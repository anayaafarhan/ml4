"""
Trains and compares models for the used-car price task, tunes the best one,
and writes the final pipeline to models/car_price_model.pkl.

Run from the project root:
    python src/train.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from data_processing import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    load_and_clean,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cardekho_used_cars.csv"
MODEL_PATH = ROOT / "models" / "car_price_model.pkl"
METADATA_PATH = ROOT / "models" / "metadata.json"
REPORT_PATH = ROOT / "reports" / "model_comparison.csv"
RANDOM_STATE = 42


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    print(f"Loading {DATA_PATH.name} ...")
    df = load_and_clean(DATA_PATH)
    print(f"{len(df)} listings left after cleaning.\n")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    results = []
    fitted_pipelines = {}

    for name, model in candidates.items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        start = time.time()
        pipe.fit(X_train, y_train)
        elapsed = time.time() - start

        preds = pipe.predict(X_test)
        metrics = evaluate(y_test, preds)
        metrics["model"] = name
        metrics["train_seconds"] = round(elapsed, 2)
        results.append(metrics)
        fitted_pipelines[name] = pipe

        print(f"{name:<20} RMSE={metrics['RMSE']:,.0f}  R2={metrics['R2']:.3f}  "
              f"({elapsed:.1f}s)")

    results_df = pd.DataFrame(results).set_index("model")
    results_df = results_df[["MAE", "MSE", "RMSE", "R2", "train_seconds"]]

    best_name = results_df["RMSE"].idxmin()
    print(f"\nBest model before tuning: {best_name}")

    # Random forest and gradient boosting both have big hyperparameter
    # spaces, so RandomizedSearchCV instead of an exhaustive grid.
    param_distributions = {
        "Random Forest": {
            "model__n_estimators": [200, 300, 400, 500],
            "model__max_depth": [None, 8, 12, 16, 20],
            "model__min_samples_split": [2, 4, 6, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4, 5],
            "model__subsample": [0.7, 0.85, 1.0],
        },
        "Decision Tree": {
            "model__max_depth": [None, 5, 10, 15, 20],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
        },
        "Linear Regression": None,  # nothing meaningful to tune here
    }

    dist = param_distributions.get(best_name)
    if dist is None:
        print(f"{best_name} has no hyperparameters worth searching, "
              "keeping it as-is.")
        tuned_pipeline = fitted_pipelines[best_name]
        tuned_metrics = results_df.loc[best_name].to_dict()
    else:
        print(f"Running RandomizedSearchCV on {best_name} ...")
        search_pipe = Pipeline([("preprocess", build_preprocessor()),
                                 ("model", candidates[best_name])])
        search = RandomizedSearchCV(
            search_pipe,
            param_distributions=dist,
            n_iter=25,
            cv=5,
            scoring="neg_root_mean_squared_error",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        tuned_pipeline = search.best_estimator_
        print(f"Best params: {search.best_params_}")

        preds = tuned_pipeline.predict(X_test)
        tuned_metrics = evaluate(y_test, preds)

    before = results_df.loc[best_name]
    print("\nBefore tuning vs after tuning:")
    print(f"  RMSE: {before['RMSE']:,.0f}  ->  {tuned_metrics['RMSE']:,.0f}")
    print(f"  R2:   {before['R2']:.3f}      ->  {tuned_metrics['R2']:.3f}")

    comparison_rows = results_df.reset_index()
    comparison_rows.loc[len(comparison_rows)] = {
        "model": f"{best_name} (tuned)",
        "MAE": tuned_metrics["MAE"],
        "MSE": tuned_metrics["MSE"],
        "RMSE": tuned_metrics["RMSE"],
        "R2": tuned_metrics["R2"],
        "train_seconds": np.nan,
    }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    comparison_rows.to_csv(REPORT_PATH, index=False)
    print(f"\nSaved comparison table to {REPORT_PATH.relative_to(ROOT)}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(tuned_pipeline, MODEL_PATH)
    print(f"Saved final model to {MODEL_PATH.relative_to(ROOT)}")

    metadata = {
        "chosen_model": f"{best_name} (tuned)" if dist is not None else best_name,
        "test_rmse": round(float(tuned_metrics["RMSE"]), 2),
        "test_r2": round(float(tuned_metrics["R2"]), 4),
        "brands": sorted(df["brand"].unique().tolist()),
        "fuel_types": sorted(df["fuel"].unique().tolist()),
        "seller_types": sorted(df["seller_type"].unique().tolist()),
        "transmissions": sorted(df["transmission"].unique().tolist()),
        "owner_types": sorted(df["owner"].unique().tolist()),
        "car_age_range": [int(df["car_age"].min()), int(df["car_age"].max())],
        "km_driven_range": [int(df["km_driven"].min()), int(df["km_driven"].max())],
        "median_price_by_brand": (
            df.groupby("brand")["selling_price"].median().round(0).to_dict()
        ),
        "trained_rows": len(df),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
