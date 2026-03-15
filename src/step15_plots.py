"""
Step 15: Create result plots.
- Predicted vs actual for best model
- Residual plot
- Feature importance for RF and XGBoost
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import (
    TEST_PATH, FEATURE_LIST_PATH,
    BEST_MODEL_PATH, RF_MODEL_PATH, XGB_MODEL_PATH,
    FIGURES_DIR
)


def create_result_plots():
    """Generate result visualizations."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    test = pd.read_csv(TEST_PATH, parse_dates=["window_start"])

    with open(FEATURE_LIST_PATH, "rb") as f:
        feature_cols = pickle.load(f)
    with open(BEST_MODEL_PATH, "rb") as f:
        best_model = pickle.load(f)
    with open(RF_MODEL_PATH, "rb") as f:
        rf_model = pickle.load(f)
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)

    X_test = test[feature_cols]
    y_test = test["next_weight"]
    preds = best_model.predict(X_test)
    residuals = y_test.values - preds

    # Predicted vs Actual
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, preds, alpha=0.5, color="steelblue", s=20)
    mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect Prediction")
    ax.set_xlabel("Actual Next Weight (g)")
    ax.set_ylabel("Predicted Next Weight (g)")
    ax.set_title("Predicted vs Actual — Best Model (XGBoost)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "predicted_vs_actual.png"))
    plt.close()

    # Residual plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(preds, residuals, alpha=0.5, color="coral", s=20)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Predicted Next Weight (g)")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residual Plot — Best Model (XGBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "residual_plot.png"))
    plt.close()

    # Feature importance — Random Forest
    rf_imp = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    rf_imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Random Forest Feature Importance (Top 20)")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "rf_feature_importance.png"))
    plt.close()

    # Feature importance — XGBoost
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb_imp.plot(kind="barh", ax=ax, color="coral")
    ax.set_title("XGBoost Feature Importance (Top 20)")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "xgb_feature_importance.png"))
    plt.close()

    print(f"Result plots saved to {FIGURES_DIR}")


if __name__ == "__main__":
    create_result_plots()
