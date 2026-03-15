"""
Step 7: Decide modeling target.
- Inspect Weight repetition
- Create next_weight (shifted) or weight_gain
- Document decision
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.config import CLEANED_DATA_PATH, TARGET_DECISION_PATH, METRICS_DIR


def decide_target():
    """Inspect Weight and decide on prediction target."""
    df = pd.read_csv(CLEANED_DATA_PATH, parse_dates=["created_at"])
    print(f"Loaded cleaned data: {df.shape}")

    weight_vc = df["Weight"].value_counts()
    top_repeated = weight_vc.head(5)
    repeat_fraction = (weight_vc > 1).sum() / len(weight_vc)

    print(f"Weight value count (top 5):\n{top_repeated}")
    print(f"Fraction of repeated Weight values: {repeat_fraction:.2%}")

    consec_dups = (df["Weight"] == df["Weight"].shift(1)).sum()
    consec_dup_pct = consec_dups / len(df) * 100
    print(f"Consecutive duplicate Weight rows: {consec_dups} ({consec_dup_pct:.1f}%)")

    df["next_weight"] = df["Weight"].shift(-1)
    df["weight_gain"] = df["next_weight"] - df["Weight"]

    chosen_target = "next_weight"
    reason = (
        "next_weight was chosen over weight_gain because:\n"
        "1. It gives an absolute target value directly interpretable by farmers.\n"
        "2. weight_gain between consecutive aggregation windows is a very small delta\n"
        "   relative to the absolute weight, making it a noisier and harder signal to learn.\n"
        "3. Predicting the next absolute weight is standard in fish growth modeling and\n"
        "   enables direct comparisons with known fish growth curves.\n"
        f"4. Consecutive duplicate Weight rows in raw data: {consec_dups} ({consec_dup_pct:.1f}%).\n"
        "   After aggregation into time windows, each window has a distinct mean Weight,\n"
        "   so predicting the next window's weight is a well-defined regression problem."
    )

    print(f"\nChosen target: {chosen_target}")
    print(reason)

    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(TARGET_DECISION_PATH, "w") as f:
        f.write("MODELING TARGET DECISION\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Chosen target column: {chosen_target}\n\n")
        f.write("Reason:\n")
        f.write(reason + "\n\n")
        f.write(f"Weight value counts (top 10):\n{weight_vc.head(10).to_string()}\n")

    print(f"Target decision saved to {TARGET_DECISION_PATH}")
    return df, chosen_target


if __name__ == "__main__":
    decide_target()
