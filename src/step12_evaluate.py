"""
Step 12: Evaluate models on validation set.
- Compute MAE, RMSE, R²
- Save validation results
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import (
    VAL_PATH, MODELS_DIR,
    LINEAR_MODEL_PATH, RF_MODEL_PATH, XGB_MODEL_PATH,
    FEATURE_LIST_PATH, SCALER_PATH, VALIDATION_RESULTS_PATH, METRICS_DIR
)


def evaluate_on_validation():
    """Evaluate all models on validation data."""
    os.makedirs(METRICS_DIR, exist_ok=True)

    val = pd.read_csv(VAL_PATH, parse_dates=["window_start"])

    with open(FEATURE_LIST_PATH, "rb") as f:
        feature_cols = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    X_val = val[feature_cols]
    y_val = val["next_weight"]
    X_val_scaled = scaler.transform(X_val)

    models = {
        "LinearRegression": (LINEAR_MODEL_PATH, True),
        "RandomForest": (RF_MODEL_PATH, False),
        "XGBoost": (XGB_MODEL_PATH, False),
    }

    results = []
    for name, (path, use_scaler) in models.items():
        with open(path, "rb") as f:
            model = pickle.load(f)
        X_input = X_val_scaled if use_scaler else X_val
        preds = model.predict(X_input)
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)
        results.append({"Model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)})
        print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(VALIDATION_RESULTS_PATH, index=False)
    print(f"\nValidation results saved to {VALIDATION_RESULTS_PATH}")
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    evaluate_on_validation()
