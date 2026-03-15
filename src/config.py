"""
Configuration file for the aquaculture ML pipeline.
All paths and parameters are controlled from here.
"""
import os

# Root directory (project root)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
INTERIM_DATA_DIR = os.path.join(ROOT_DIR, "data", "interim")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")

# Model paths
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Report paths
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
FIGURES_DIR = os.path.join(ROOT_DIR, "reports", "figures")
LOGS_DIR = os.path.join(ROOT_DIR, "reports", "logs")
METRICS_DIR = os.path.join(ROOT_DIR, "reports", "metrics")

# Interim file paths
MERGED_RAW_PATH = os.path.join(INTERIM_DATA_DIR, "merged_raw.csv")
MERGED_STANDARDIZED_PATH = os.path.join(INTERIM_DATA_DIR, "merged_standardized.csv")
MERGED_SORTED_PATH = os.path.join(INTERIM_DATA_DIR, "merged_sorted.csv")
CLEANED_DATA_PATH = os.path.join(INTERIM_DATA_DIR, "cleaned_data.csv")

# Processed file paths
AGGREGATED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "aggregated_data.csv")
MODEL_READY_PATH = os.path.join(PROCESSED_DATA_DIR, "model_ready.csv")
TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train.csv")
VAL_PATH = os.path.join(PROCESSED_DATA_DIR, "val.csv")
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test.csv")

# Report file paths
FILE_SUMMARY_PATH = os.path.join(LOGS_DIR, "file_summary.txt")
EDA_SUMMARY_PATH = os.path.join(METRICS_DIR, "eda_summary.txt")
TARGET_DECISION_PATH = os.path.join(METRICS_DIR, "target_decision.txt")
SPLIT_SUMMARY_PATH = os.path.join(METRICS_DIR, "split_summary.txt")
VALIDATION_RESULTS_PATH = os.path.join(METRICS_DIR, "validation_results.csv")
BEST_PARAMS_PATH = os.path.join(METRICS_DIR, "best_params.json")
TEST_RESULTS_PATH = os.path.join(METRICS_DIR, "test_results.json")

# Model artifact paths
LINEAR_MODEL_PATH = os.path.join(MODELS_DIR, "linear_regression.pkl")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost.pkl")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "feature_list.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Schema configuration
EXPECTED_COLUMNS = [
    "created_at",
    "entry_id",
    "TEMPERATURE",
    "TURBIDITY",
    "DISSOLVED OXYGEN",
    "pH",
    "AMMONIA",
    "NITRATE",
    "Population",
    "Length",
    "Weight",
    "source_file",
]

SENSOR_COLUMNS = [
    "TEMPERATURE",
    "TURBIDITY",
    "DISSOLVED OXYGEN",
    "pH",
    "AMMONIA",
    "NITRATE",
    "Population",
]

TARGET_COLUMN = "next_weight"  # Will be set in step07

# Aggregation window size (pandas offset string)
AGGREGATION_WINDOW = "6h"

# Lag and rolling window sizes
LAG_PERIODS = [1, 2, 3]
ROLLING_WINDOWS = [3, 6]

# Train/val/test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# XGBoost default parameters
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# Random seed
RANDOM_SEED = 42
