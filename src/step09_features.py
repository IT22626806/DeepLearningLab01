"""
Step 9: Create temporal features (lag and rolling).
- Lag features for major sensors
- Rolling mean, min, max, std
- Drop rows without enough history
- Save to data/processed/model_ready.csv
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.config import (
    AGGREGATED_DATA_PATH, MODEL_READY_PATH, PROCESSED_DATA_DIR,
    SENSOR_COLUMNS, LAG_PERIODS, ROLLING_WINDOWS
)


def create_features():
    """Generate lag and rolling features for modeling."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    df = pd.read_csv(AGGREGATED_DATA_PATH, parse_dates=["window_start"])
    df = df.sort_values("window_start").reset_index(drop=True)
    print(f"Loaded aggregated data: {df.shape}")

    # Fill NaN in _std columns with 0 (single-row windows have no std)
    std_cols = [c for c in df.columns if c.endswith("_std")]
    df[std_cols] = df[std_cols].fillna(0)

    # Base feature columns: sensor means from aggregated data
    base_feature_cols = [c for c in df.columns if c.endswith("_mean") and not c.startswith("next_")]
    for col in ["Length_mean", "Weight_mean"]:
        if col in df.columns and col not in base_feature_cols:
            base_feature_cols.append(col)

    feature_df = df.copy()

    # Lag features
    for col in base_feature_cols:
        for lag in LAG_PERIODS:
            feature_df[f"{col}_lag{lag}"] = feature_df[col].shift(lag)

    # Rolling features (shift by 1 to avoid leakage)
    for col in base_feature_cols:
        for window in ROLLING_WINDOWS:
            rolled = feature_df[col].shift(1).rolling(window)
            feature_df[f"{col}_roll{window}_mean"] = rolled.mean()
            feature_df[f"{col}_roll{window}_min"] = rolled.min()
            feature_df[f"{col}_roll{window}_max"] = rolled.max()
            feature_df[f"{col}_roll{window}_std"] = rolled.std()

    # Calendar features — build separately and concat to avoid fragmentation warning
    calendar = pd.DataFrame({
        "hour": feature_df["window_start"].dt.hour,
        "day_of_week": feature_df["window_start"].dt.dayofweek,
        "month": feature_df["window_start"].dt.month,
    }, index=feature_df.index)
    feature_df = pd.concat([feature_df, calendar], axis=1)

    before = len(feature_df)
    feature_df = feature_df.dropna(subset=["next_weight"])
    lag_roll_cols = [c for c in feature_df.columns if "_lag" in c or "_roll" in c]
    feature_df = feature_df.dropna(how="any", subset=lag_roll_cols)
    after = len(feature_df)

    print(f"Dropped {before - after} rows due to insufficient history.")
    print(f"Final model-ready shape: {feature_df.shape}")

    feature_df.to_csv(MODEL_READY_PATH, index=False)
    print(f"Saved model-ready data to {MODEL_READY_PATH}")
    return feature_df


if __name__ == "__main__":
    create_features()
