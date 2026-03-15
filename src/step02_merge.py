"""
Step 2: Load and merge all CSV files from data/raw/.
- Add source_file column
- Merge into one dataframe
- Save to data/interim/merged_raw.csv
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.config import RAW_DATA_DIR, INTERIM_DATA_DIR, MERGED_RAW_PATH


def load_and_merge():
    """Load all CSVs and merge into one dataframe."""
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")])
    print(f"Loading {len(csv_files)} CSV files...")

    dfs = []
    for fname in csv_files:
        fpath = os.path.join(RAW_DATA_DIR, fname)
        try:
            df = pd.read_csv(fpath)
            df["source_file"] = fname
            dfs.append(df)
            print(f"  Loaded {fname}: {df.shape[0]} rows, {df.shape[1]} cols")
        except Exception as e:
            print(f"  WARNING: Could not load {fname}: {e}")

    if not dfs:
        raise RuntimeError("No CSV files could be loaded.")

    merged = pd.concat(dfs, ignore_index=True, sort=False)

    print(f"\nMerge Summary:")
    print(f"  Total files loaded: {len(dfs)}")
    print(f"  Total merged rows:  {merged.shape[0]}")
    print(f"  Total merged cols:  {merged.shape[1]}")

    merged.to_csv(MERGED_RAW_PATH, index=False)
    print(f"\nSaved merged raw data to {MERGED_RAW_PATH}")
    return merged


if __name__ == "__main__":
    load_and_merge()
