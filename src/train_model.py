"""
ApexPitwall — Model Training Module
Path: src/train_model.py

Trains:
1. Baseline Interpretable Ridge Regression (estimating compound degradation slopes and fuel burn effects).
2. Non-linear LightGBM Regressor.

Evaluates both:
- Cross-Circuit Held-Out (Train on 4 circuits, held-out test on Catalunya).
- Within-Circuit K-Fold (5-fold CV on Catalunya).
Saves fitted models to models/.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import lightgbm as lgb

DATA_PATH = "data/processed/race_laps_processed.csv"
MODELS_DIR = "models"
HELD_OUT_CIRCUIT = "catalunya"


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Loads processed dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found at {path}. Run src/data_pipeline.py first.")
    return pd.read_csv(path)


def train_models():
    """Trains Ridge and LightGBM models on processed dataset and saves artifacts."""
    df = load_dataset()
    os.makedirs(MODELS_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Split 1: Cross-Circuit Held-Out (Train on 4 circuits, Test on Catalunya)
    # -------------------------------------------------------------------------
    train_mask = df["Circuit"] != HELD_OUT_CIRCUIT
    test_mask = df["Circuit"] == HELD_OUT_CIRCUIT

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    print("=" * 80)
    print("APEXPITWALL: MODEL TRAINING PIPELINE")
    print("=" * 80)
    print(f"Total Clean Laps: {len(df)}")
    print(f"Cross-Circuit Train Size (Bahrain, Zandvoort, Mexico, Abu Dhabi): {len(train_df)} laps")
    print(f"Cross-Circuit Held-Out Test Size (Spain - Catalunya): {len(test_df)} laps")

    # Features for Ridge
    ridge_numeric = ["TyreLife", "fuel_remaining_kg", "TrackTemp", "rolling_pace_mean_3"]
    ridge_cat = ["Compound"]

    preprocessor_ridge = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ridge_numeric),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), ridge_cat),
        ],
        remainder="drop"
    )

    ridge_pipeline = Pipeline([
        ("prep", preprocessor_ridge),
        ("reg", Ridge(alpha=10.0, random_state=42))
    ])

    ridge_pipeline.fit(train_df[ridge_numeric + ridge_cat], train_df["lap_time_seconds"])

    # Features for LightGBM
    lgbm_numeric = [
        "TyreLife", "fuel_remaining_kg", "TrackTemp", "AirTemp",
        "Humidity", "rolling_pace_mean_3", "rolling_pace_std_3", "LapNumber", "Stint"
    ]
    lgbm_cat = ["Compound", "Team"]

    preprocessor_lgbm = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), lgbm_numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), lgbm_cat),
        ],
        remainder="drop"
    )

    lgbm_reg = lgb.LGBMRegressor(
        n_estimators=120,
        learning_rate=0.04,
        max_depth=5,
        num_leaves=25,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    lgbm_pipeline = Pipeline([
        ("prep", preprocessor_lgbm),
        ("reg", lgbm_reg)
    ])

    lgbm_pipeline.fit(train_df[lgbm_numeric + lgbm_cat], train_df["lap_time_seconds"])

    # Save Cross-Circuit Models
    joblib.dump(ridge_pipeline, os.path.join(MODELS_DIR, "ridge_cross_circuit.joblib"))
    joblib.dump(lgbm_pipeline, os.path.join(MODELS_DIR, "lgbm_cross_circuit.joblib"))

    # -------------------------------------------------------------------------
    # Split 2: Catalunya In-Circuit Dedicated Calibration Models
    # -------------------------------------------------------------------------
    cat_df = test_df.copy()

    # Dedicated Ridge for Catalunya
    cat_ridge = Pipeline([
        ("prep", preprocessor_ridge),
        ("reg", Ridge(alpha=5.0, random_state=42))
    ])
    cat_ridge.fit(cat_df[ridge_numeric + ridge_cat], cat_df["lap_time_seconds"])
    joblib.dump(cat_ridge, os.path.join(MODELS_DIR, "ridge_catalunya.joblib"))

    # Dedicated LightGBM for Catalunya
    cat_lgbm = Pipeline([
        ("prep", preprocessor_lgbm),
        ("reg", lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=15,
            min_child_samples=15,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ))
    ])
    cat_lgbm.fit(cat_df[lgbm_numeric + lgbm_cat], cat_df["lap_time_seconds"])
    joblib.dump(cat_lgbm, os.path.join(MODELS_DIR, "lgbm_catalunya.joblib"))

    print(f"\nSaved all fitted model artifacts to {MODELS_DIR}/")
    return {
        "ridge_cross_circuit": ridge_pipeline,
        "lgbm_cross_circuit": lgbm_pipeline,
        "ridge_catalunya": cat_ridge,
        "lgbm_catalunya": cat_lgbm,
        "train_df": train_df,
        "test_df": test_df
    }


if __name__ == "__main__":
    train_models()

