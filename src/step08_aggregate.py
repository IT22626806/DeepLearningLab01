"""
Step 8: Aggregate repeated sensor rows into time windows.
- Window size configurable in src/config.py
- Compute mean, min, max, std for each sensor
- Keep representative target values
- Save to data/processed/aggregated_data.csv
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.config import (
    CLEANED_DATA_PATH, AGGREGATED_DATA_PATH, PROCESSED_DATA_DIR,
    SENSOR_COLUMNS, AGGREGATION_WINDOW
)


def aggregate_data():
    """Aggregate sensor data into time windows."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    df = pd.read_csv(CLEANED_DATA_PATH, parse_dates=["created_at"])
    print(f"Loaded cleaned data: {df.shape}")
    print(f"Aggregation window: {AGGREGATION_WINDOW}")

    df = df.set_index("created_at").sort_index()

    available_sensors = [c for c in SENSOR_COLUMNS if c in df.columns]
    target_cols = [c for c in ["Length", "Weight"] if c in df.columns]

    sensor_agg = {col: ["mean", "min", "max", "std"] for col in available_sensors}
    for col in target_cols:
        sensor_agg[col] = "mean"

    agg_df = df[available_sensors + target_cols].resample(AGGREGATION_WINDOW).agg(sensor_agg)

    # Flatten multi-level columns
    agg_df.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in agg_df.columns
    ]

    agg_df = agg_df.dropna(how="all").reset_index()
    agg_df = agg_df.rename(columns={"created_at": "window_start"})

    if "Weight_mean" in agg_df.columns:
        agg_df["next_weight"] = agg_df["Weight_mean"].shift(-1)
        agg_df["weight_gain"] = agg_df["next_weight"] - agg_df["Weight_mean"]

    print(f"Aggregated shape: {agg_df.shape}")
    print(f"Columns: {list(agg_df.columns)}")

    agg_df.to_csv(AGGREGATED_DATA_PATH, index=False)
    print(f"Saved aggregated data to {AGGREGATED_DATA_PATH}")
    return agg_df


if __name__ == "__main__":
    aggregate_data()
