"""
Step 5: Basic cleaning.
- Remove exact duplicate rows
- Check missing values for every column
- Create missing value report
- Save to data/interim/cleaned_data.csv
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.config import MERGED_SORTED_PATH, CLEANED_DATA_PATH, LOGS_DIR


def basic_cleaning():
    """Remove duplicates and report missing values."""
    df = pd.read_csv(MERGED_SORTED_PATH)
    print(f"Loaded sorted data: {df.shape}")

    before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before - len(df)
    print(f"Removed {duplicates_removed} exact duplicate rows.")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})

    print("\nMissing Value Report:")
    print(missing_report[missing_report["missing_count"] > 0])

    os.makedirs(LOGS_DIR, exist_ok=True)
    missing_report_path = os.path.join(LOGS_DIR, "missing_value_report.txt")
    with open(missing_report_path, "w") as f:
        f.write("MISSING VALUE REPORT\n")
        f.write("=" * 40 + "\n")
        f.write(missing_report.to_string())
        f.write(f"\n\nDuplicate rows removed: {duplicates_removed}\n")
    print(f"Missing value report saved to {missing_report_path}")

    os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Saved cleaned data to {CLEANED_DATA_PATH}")
    return df


if __name__ == "__main__":
    basic_cleaning()
