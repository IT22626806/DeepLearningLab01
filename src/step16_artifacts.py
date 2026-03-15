"""
Step 16: Save final artifacts.
- best_model.pkl (already saved in step14)
- feature_list.pkl (already saved in step11)
- scaler.pkl (already saved in step11)
- Save inference sample input/output
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import json
import pandas as pd
from src.config import (
    TEST_PATH, FEATURE_LIST_PATH, BEST_MODEL_PATH, MODELS_DIR, SCALER_PATH
)


def save_artifacts():
    """Confirm and document all saved artifacts."""
    artifacts = [
        ("Best Model", BEST_MODEL_PATH),
        ("Feature List", FEATURE_LIST_PATH),
        ("Scaler", SCALER_PATH),
    ]

    print("=== Final Artifacts ===")
    for name, path in artifacts:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"  {name}: {path} ({'OK' if exists else 'MISSING'}, {size} bytes)")

    # Save sample inference input/output
    test = pd.read_csv(TEST_PATH)
    with open(FEATURE_LIST_PATH, "rb") as f:
        feature_cols = pickle.load(f)
    with open(BEST_MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    sample_input = test[feature_cols].iloc[:5]
    sample_output = model.predict(sample_input)

    sample_input_path = os.path.join(MODELS_DIR, "sample_input.csv")
    sample_output_path = os.path.join(MODELS_DIR, "sample_output.json")

    sample_input.to_csv(sample_input_path, index=False)
    with open(sample_output_path, "w") as f:
        json.dump({"predicted_next_weight": [float(v) for v in sample_output.round(4)]}, f, indent=2)

    print(f"\nSample input saved to {sample_input_path}")
    print(f"Sample output saved to {sample_output_path}")

    artifact_list_path = os.path.join(MODELS_DIR, "artifact_manifest.json")
    manifest = {
        "best_model": BEST_MODEL_PATH,
        "feature_list": FEATURE_LIST_PATH,
        "scaler": SCALER_PATH,
        "sample_input": sample_input_path,
        "sample_output": sample_output_path,
    }
    with open(artifact_list_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Artifact manifest saved to {artifact_list_path}")

    return manifest


if __name__ == "__main__":
    save_artifacts()
