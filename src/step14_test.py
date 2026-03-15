"""
Step 14: Final test evaluation using best model.
- Compare all model validation results
- Select the model with best validation R²
- Retrain best model on train+val combined
- Evaluate on held-out test set
- Save test metrics to reports/metrics/test_results.json
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from src.config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    FEATURE_LIST_PATH, BEST_PARAMS_PATH, BEST_MODEL_PATH,
    VALIDATION_RESULTS_PATH, SCALER_PATH,
    TEST_RESULTS_PATH, METRICS_DIR, MODELS_DIR, RANDOM_SEED
)


def final_test():
    """Select best model from validation results and evaluate on test set."""
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH, parse_dates=["window_start"])
    val = pd.read_csv(VAL_PATH, parse_dates=["window_start"])
    test = pd.read_csv(TEST_PATH, parse_dates=["window_start"])

    with open(FEATURE_LIST_PATH, "rb") as f:
        feature_cols = pickle.load(f)
    with open(BEST_PARAMS_PATH, "r") as f:
        best_params = json.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Select best model based on validation R²
    val_results = pd.read_csv(VALIDATION_RESULTS_PATH)
    best_row = val_results.loc[val_results["R2"].idxmax()]
    best_model_name = best_row["Model"]
    print(f"Best model by validation R²: {best_model_name} (R²={best_row['R2']:.4f})")

    # Combine train + val for final training
    train_val = pd.concat([train, val], ignore_index=True)
    X_train_val = train_val[feature_cols]
    y_train_val = train_val["next_weight"]
    X_test = test[feature_cols]
    y_test = test["next_weight"]

    # Instantiate and train the chosen best model
    print(f"Training best model ({best_model_name}) on {len(X_train_val)} rows (train+val)...")
    if best_model_name == "LinearRegression":
        X_train_val_scaled = scaler.fit_transform(X_train_val)
        X_test_scaled = scaler.transform(X_test)
        best_model = LinearRegression()
        best_model.fit(X_train_val_scaled, y_train_val)
        preds = best_model.predict(X_test_scaled)
    elif best_model_name == "RandomForest":
        best_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
        best_model.fit(X_train_val, y_train_val)
        preds = best_model.predict(X_test)
    else:
        best_model = XGBRegressor(**best_params)
        best_model.fit(X_train_val, y_train_val)
        preds = best_model.predict(X_test)

    # Save best model
    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Best model saved to {BEST_MODEL_PATH}")

    # Evaluate on test set
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results = {
        "best_model": best_model_name,
        "test_MAE": round(float(mae), 4),
        "test_RMSE": round(float(rmse), 4),
        "test_R2": round(float(r2), 4),
    }
    print(f"\nTest Results: {results}")

    with open(TEST_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Test results saved to {TEST_RESULTS_PATH}")

    return best_model, preds, y_test


if __name__ == "__main__":
    final_test()
