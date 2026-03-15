"""
Step 3: Standardize schema across merged dataframe.
- Normalize column names (strip spaces, fix typos)
- Fix Lenght -> Length
- Ensure expected columns are present
- Save to data/interim/merged_standardized.csv
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.config import MERGED_RAW_PATH, MERGED_STANDARDIZED_PATH, EXPECTED_COLUMNS

# Map known typos/variants to correct names
COLUMN_RENAME_MAP = {
    "Lenght": "Length",
    "lenght": "Length",
    "length": "Length",
    "weight": "Weight",
    "temperature": "TEMPERATURE",
    "turbidity": "TURBIDITY",
    "dissolved oxygen": "DISSOLVED OXYGEN",
    "ph": "pH",
    "ammonia": "AMMONIA",
    "nitrate": "NITRATE",
    "population": "Population",
}


def standardize_schema():
    """Standardize column names and fix mismatches."""
    df = pd.read_csv(MERGED_RAW_PATH)
    print(f"Loaded merged raw data: {df.shape}")

    original_cols = list(df.columns)

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Apply rename map
    rename_dict = {col: COLUMN_RENAME_MAP[col] for col in df.columns if col in COLUMN_RENAME_MAP}

    if rename_dict:
        print(f"Renaming columns: {rename_dict}")
        df = df.rename(columns=rename_dict)

    # Coalesce duplicate columns that arose from renaming (e.g., two 'Length' columns)
    # For each duplicated column name, combine all occurrences by taking the first non-null value
    if df.columns.duplicated().any():
        deduped_cols = {}
        for col in df.columns.unique():
            col_data = df.loc[:, df.columns == col]
            if col_data.shape[1] > 1:
                # Coalesce: first non-null value across duplicate columns
                deduped_cols[col] = col_data.bfill(axis=1).iloc[:, 0]
            else:
                deduped_cols[col] = col_data.iloc[:, 0]
        df = pd.DataFrame(deduped_cols)

    final_cols = list(df.columns)
    print(f"Original columns: {original_cols}")
    print(f"Final columns:    {final_cols}")

    missing_expected = [c for c in EXPECTED_COLUMNS if c not in final_cols]
    if missing_expected:
        print(f"WARNING: Missing expected columns: {missing_expected}")
    else:
        print("All expected columns present.")

    os.makedirs(os.path.dirname(MERGED_STANDARDIZED_PATH), exist_ok=True)
    df.to_csv(MERGED_STANDARDIZED_PATH, index=False)
    print(f"Saved standardized data to {MERGED_STANDARDIZED_PATH}")
    return df


if __name__ == "__main__":
    standardize_schema()
