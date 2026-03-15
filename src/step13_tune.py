"""
Step 13: Tune XGBoost on validation set.
- Grid search over key hyperparameters
- Save best params to reports/metrics/best_params.json
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from src.config import (
    TRAIN_PATH, VAL_PATH, FEATURE_LIST_PATH,
    BEST_PARAMS_PATH, METRICS_DIR, RANDOM_SEED
)


def tune_xgboost():
    """Tune XGBoost hyperparameters using the validation set."""
    os.makedirs(METRICS_DIR, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH, parse_dates=["window_start"])
    val = pd.read_csv(VAL_PATH, parse_dates=["window_start"])

    with open(FEATURE_LIST_PATH, "rb") as f:
        feature_cols = pickle.load(f)

    X_train = train[feature_cols]
    y_train = train["next_weight"]
    X_val = val[feature_cols]
    y_val = val["next_weight"]

    param_grid = [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": s, "colsample_bytree": c}
        for n in [100, 200]
        for d in [3, 5]
        for lr in [0.05, 0.1]
        for s in [0.8]
        for c in [0.8]
    ]

    best_rmse = float("inf")
    best_params = None

    print(f"Tuning over {len(param_grid)} configurations...")
    for i, params in enumerate(param_grid):
        xgb = XGBRegressor(**params, random_state=RANDOM_SEED, verbosity=0)
        xgb.fit(X_train, y_train)
        preds = xgb.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
        if (i + 1) % 4 == 0:
            print(f"  [{i+1}/{len(param_grid)}] Best RMSE so far: {best_rmse:.4f}")

    best_params["random_state"] = RANDOM_SEED
    print(f"\nBest params: {best_params}")
    print(f"Best validation RMSE: {best_rmse:.4f}")

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"Best params saved to {BEST_PARAMS_PATH}")
    return best_params


if __name__ == "__main__":
    tune_xgboost()
