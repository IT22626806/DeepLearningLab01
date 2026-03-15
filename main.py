"""
Main entry point for the aquaculture fish farming ML pipeline.

Runs all 16 steps sequentially:
  Step 01 — Inspect raw CSV files
  Step 02 — Load and merge CSVs
  Step 03 — Standardize schema
  Step 04 — Parse datetime and sort
  Step 05 — Basic cleaning
  Step 06 — Exploratory Data Analysis
  Step 07 — Decide modeling target
  Step 08 — Aggregate sensor rows
  Step 09 — Create temporal features
  Step 10 — Train/val/test split
  Step 11 — Train baseline models
  Step 12 — Evaluate on validation set
  Step 13 — Tune XGBoost
  Step 14 — Final test evaluation
  Step 15 — Generate result plots
  Step 16 — Save final artifacts

Usage:
    python main.py
"""
import sys
import os
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_step(step_name: str, fn):
    """Run a single pipeline step with timing and error handling."""
    print(f"\n{'=' * 60}")
    print(f"  {step_name}")
    print(f"{'=' * 60}")
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"  ✓ {step_name} completed in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ✗ {step_name} FAILED after {elapsed:.1f}s: {e}")
        raise


def main():
    from src.step01_inspect import inspect_raw_files
    from src.step02_merge import load_and_merge
    from src.step03_standardize import standardize_schema
    from src.step04_datetime import parse_and_sort
    from src.step05_clean import basic_cleaning
    from src.step06_eda import run_eda
    from src.step07_target import decide_target
    from src.step08_aggregate import aggregate_data
    from src.step09_features import create_features
    from src.step10_split import split_data
    from src.step11_train import train_models
    from src.step12_evaluate import evaluate_on_validation
    from src.step13_tune import tune_xgboost
    from src.step14_test import final_test
    from src.step15_plots import create_result_plots
    from src.step16_artifacts import save_artifacts

    pipeline = [
        ("Step 01 — Inspect raw CSV files",     inspect_raw_files),
        ("Step 02 — Load and merge CSVs",        load_and_merge),
        ("Step 03 — Standardize schema",         standardize_schema),
        ("Step 04 — Parse datetime and sort",    parse_and_sort),
        ("Step 05 — Basic cleaning",             basic_cleaning),
        ("Step 06 — Exploratory Data Analysis",  run_eda),
        ("Step 07 — Decide modeling target",     decide_target),
        ("Step 08 — Aggregate sensor rows",      aggregate_data),
        ("Step 09 — Create temporal features",   create_features),
        ("Step 10 — Train/val/test split",       split_data),
        ("Step 11 — Train baseline models",      train_models),
        ("Step 12 — Evaluate on validation",     evaluate_on_validation),
        ("Step 13 — Tune XGBoost",               tune_xgboost),
        ("Step 14 — Final test evaluation",      final_test),
        ("Step 15 — Generate result plots",      create_result_plots),
        ("Step 16 — Save final artifacts",       save_artifacts),
    ]

    total_start = time.time()
    print("\n" + "=" * 60)
    print("  AQUACULTURE ML PIPELINE — START")
    print("=" * 60)

    for step_name, fn in pipeline:
        run_step(step_name, fn)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE — {total_elapsed:.1f}s total")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
