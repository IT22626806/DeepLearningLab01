"""
Step 1: Inspect raw CSV files in data/raw/.
- Detect all CSV files
- List file names
- Extract numeric parts from filenames
- Report missing numbered files
- Save file summary report
"""
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.config import RAW_DATA_DIR, LOGS_DIR, FILE_SUMMARY_PATH


def inspect_raw_files():
    """Inspect all CSV files in the raw data directory."""
    os.makedirs(LOGS_DIR, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")])
    print(f"Found {len(csv_files)} CSV files in {RAW_DATA_DIR}")
    print("Files:", csv_files)

    numeric_parts = []
    for fname in csv_files:
        nums = re.findall(r"\d+", fname)
        if nums:
            numeric_parts.extend([int(n) for n in nums])

    missing = []
    if numeric_parts:
        numeric_parts = sorted(numeric_parts)
        expected = list(range(min(numeric_parts), max(numeric_parts) + 1))
        missing = [n for n in expected if n not in numeric_parts]
        if missing:
            print(f"WARNING: Missing numbered files: {missing}")
        else:
            print(f"No missing numbered files detected (range {min(numeric_parts)}-{max(numeric_parts)})")

    summary_lines = ["=" * 60, "FILE SUMMARY REPORT", "=" * 60, ""]
    summary_lines.append(f"Total CSV files found: {len(csv_files)}")
    summary_lines.append(f"Directory: {RAW_DATA_DIR}")
    summary_lines.append("")

    file_info = []
    for fname in csv_files:
        fpath = os.path.join(RAW_DATA_DIR, fname)
        try:
            df = pd.read_csv(fpath)
            rows, cols = df.shape
            col_names = list(df.columns)
            file_info.append({"filename": fname, "rows": rows, "columns": col_names})
            summary_lines.append(f"File: {fname}")
            summary_lines.append(f"  Rows: {rows}")
            summary_lines.append(f"  Columns ({cols}): {col_names}")
            summary_lines.append("")
        except Exception as e:
            summary_lines.append(f"File: {fname} — ERROR reading: {e}")
            summary_lines.append("")

    if numeric_parts and missing:
        summary_lines.append(f"WARNING: Missing numbered files: {missing}")
        summary_lines.append("")

    report_text = "\n".join(summary_lines)
    with open(FILE_SUMMARY_PATH, "w") as f:
        f.write(report_text)

    print(f"File summary saved to {FILE_SUMMARY_PATH}")
    return csv_files, file_info


if __name__ == "__main__":
    inspect_raw_files()
