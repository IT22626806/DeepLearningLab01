"""
Step 10: Chronological train/validation/test split.
- No random split
- 70% train, 15% val, 15% test
- Save split files and summary
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.config import (
    MODEL_READY_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH,
    SPLIT_SUMMARY_PATH, PROCESSED_DATA_DIR, METRICS_DIR,
    TRAIN_RATIO, VAL_RATIO
)


def split_data():
    """Split data chronologically."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    df = pd.read_csv(MODEL_READY_PATH, parse_dates=["window_start"])
    df = df.sort_values("window_start").reset_index(drop=True)
    n = len(df)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    train.to_csv(TRAIN_PATH, index=False)
    val.to_csv(VAL_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    summary = (
        f"SPLIT SUMMARY\n"
        f"{'=' * 40}\n"
        f"Total rows: {n}\n"
        f"Train rows: {len(train)} ({len(train)/n*100:.1f}%) "
        f"— {train['window_start'].min()} to {train['window_start'].max()}\n"
        f"Val rows:   {len(val)} ({len(val)/n*100:.1f}%) "
        f"— {val['window_start'].min()} to {val['window_start'].max()}\n"
        f"Test rows:  {len(test)} ({len(test)/n*100:.1f}%) "
        f"— {test['window_start'].min()} to {test['window_start'].max()}\n"
        f"\nSplit method: Chronological (no random shuffling)\n"
        f"Train ratio: {TRAIN_RATIO}, Val ratio: {VAL_RATIO}\n"
    )

    print(summary)
    with open(SPLIT_SUMMARY_PATH, "w") as f:
        f.write(summary)

    print(f"Saved train/val/test to {PROCESSED_DATA_DIR}")
    print(f"Saved split summary to {SPLIT_SUMMARY_PATH}")
    return train, val, test


if __name__ == "__main__":
    split_data()
