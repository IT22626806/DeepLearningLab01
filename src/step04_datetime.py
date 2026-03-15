"""
Step 4: Parse datetime and sort chronologically.
- Convert created_at to datetime
- Handle timezone strings safely
- Drop invalid rows and log count
- Sort by created_at
- Save to data/interim/merged_sorted.csv
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.config import MERGED_STANDARDIZED_PATH, MERGED_SORTED_PATH


def parse_and_sort():
    """Parse datetime column and sort rows chronologically."""
    df = pd.read_csv(MERGED_STANDARDIZED_PATH)
    print(f"Loaded standardized data: {df.shape}")

    original_count = len(df)

    def safe_parse(val):
        try:
            return pd.to_datetime(str(val).strip(), utc=True)
        except Exception:
            return pd.NaT

    df["created_at"] = df["created_at"].apply(safe_parse)

    invalid_count = df["created_at"].isna().sum()
    if invalid_count > 0:
        print(f"Dropping {invalid_count} rows with invalid datetime.")
        df = df.dropna(subset=["created_at"])
    else:
        print("No invalid datetime rows found.")

    dropped = original_count - len(df)
    print(f"Rows dropped due to invalid datetime: {dropped}")

    df = df.sort_values("created_at").reset_index(drop=True)
    print(f"Sorted {len(df)} rows chronologically.")

    os.makedirs(os.path.dirname(MERGED_SORTED_PATH), exist_ok=True)
    df.to_csv(MERGED_SORTED_PATH, index=False)
    print(f"Saved sorted data to {MERGED_SORTED_PATH}")
    return df


if __name__ == "__main__":
    parse_and_sort()
