"""
Step 11: Train baseline models.
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- Save trained models to models/
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from src.config import (
    TRAIN_PATH, MODELS_DIR,
    LINEAR_MODEL_PATH, RF_MODEL_PATH, XGB_MODEL_PATH,
    FEATURE_LIST_PATH, SCALER_PATH, XGB_PARAMS, RANDOM_SEED
)

NON_FEATURE_COLS = ["window_start", "next_weight", "weight_gain", "Weight_mean", "Length_mean"]


def get_features_and_target(df):
    """Extract feature matrix and target vector."""
    exclude = [c for c in NON_FEATURE_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].select_dtypes(include="number")
    y = df["next_weight"]
    return X, y


def train_models():
    """Train all baseline models."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH, parse_dates=["window_start"])
    X_train, y_train = get_features_and_target(train)

    feature_cols = list(X_train.columns)
    print(f"Training on {len(X_train)} samples with {len(feature_cols)} features.")

    with open(FEATURE_LIST_PATH, "wb") as f:
        pickle.dump(feature_cols, f)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    with open(LINEAR_MODEL_PATH, "wb") as f:
        pickle.dump(lr, f)
    print(f"  Saved to {LINEAR_MODEL_PATH}")

    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    with open(RF_MODEL_PATH, "wb") as f:
        pickle.dump(rf, f)
    print(f"  Saved to {RF_MODEL_PATH}")

    print("Training XGBoost...")
    xgb = XGBRegressor(**XGB_PARAMS)
    xgb.fit(X_train, y_train)
    with open(XGB_MODEL_PATH, "wb") as f:
        pickle.dump(xgb, f)
    print(f"  Saved to {XGB_MODEL_PATH}")

    print("\nAll models trained and saved.")
    return lr, rf, xgb, scaler, feature_cols


if __name__ == "__main__":
    train_models()
