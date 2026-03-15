"""
Inference script.
- Load best model and feature list
- Accept a processed input CSV file
- Return predictions saved to CSV

Usage:
    python src/predict.py --input <path_to_input.csv> --output <path_to_output.csv>
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
import pandas as pd
from src.config import BEST_MODEL_PATH, FEATURE_LIST_PATH


def predict(input_path: str, output_path: str):
    """Load model and generate predictions."""
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Best model not found at {BEST_MODEL_PATH}. Run the full pipeline first."
        )
    if not os.path.exists(FEATURE_LIST_PATH):
        raise FileNotFoundError(
            f"Feature list not found at {FEATURE_LIST_PATH}. Run the full pipeline first."
        )

    with open(BEST_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURE_LIST_PATH, "rb") as f:
        feature_cols = pickle.load(f)

    df = pd.read_csv(input_path)

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input file is missing required columns: {missing_cols}")

    X = df[feature_cols]
    preds = model.predict(X)

    output_df = df.copy()
    output_df["predicted_next_weight"] = preds.round(4)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"Sample predictions: {preds[:5].round(4)}")
    return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with best model.")
    parser.add_argument("--input", required=True, help="Path to input CSV file (model-ready features).")
    parser.add_argument("--output", default="predictions.csv", help="Path to save predictions CSV.")
    args = parser.parse_args()
    predict(args.input, args.output)
