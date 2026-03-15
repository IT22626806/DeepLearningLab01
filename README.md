# Aquaculture Fish Farming — ML Pipeline

A complete, reproducible data science pipeline for predicting fish weight from aquaculture sensor data. Raw IoT sensor readings are ingested, cleaned, feature-engineered, and used to train regression models that forecast the **next measured fish weight** (`next_weight`).

---

## Project Structure

```
.
├── data/
│   ├── raw/                   # Original CSV files (feed1.csv, feed2.csv, feed3.csv)
│   ├── interim/               # Intermediate processed files
│   │   ├── merged_raw.csv
│   │   ├── merged_standardized.csv
│   │   ├── merged_sorted.csv
│   │   └── cleaned_data.csv
│   └── processed/             # Final model-ready datasets
│       ├── aggregated_data.csv
│       ├── model_ready.csv
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── models/                    # Saved model artifacts
│   ├── best_model.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── linear_regression.pkl
│   ├── feature_list.pkl
│   ├── scaler.pkl
│   ├── sample_input.csv
│   ├── sample_output.json
│   └── artifact_manifest.json
├── reports/
│   ├── figures/               # All plots (PNG)
│   ├── logs/                  # File inspection and missing-value reports
│   └── metrics/               # EDA summary, split info, validation/test results
├── src/
│   ├── config.py              # All paths and hyperparameters
│   ├── step01_inspect.py      # Inspect raw files
│   ├── step02_merge.py        # Load and merge CSVs
│   ├── step03_standardize.py  # Fix schema and column name typos
│   ├── step04_datetime.py     # Parse datetime, sort chronologically
│   ├── step05_clean.py        # Remove duplicates, report missing values
│   ├── step06_eda.py          # Exploratory Data Analysis + plots
│   ├── step07_target.py       # Decide modeling target
│   ├── step08_aggregate.py    # Aggregate sensor rows into time windows
│   ├── step09_features.py     # Lag and rolling temporal features
│   ├── step10_split.py        # Chronological train/val/test split
│   ├── step11_train.py        # Train Linear, RF, XGBoost models
│   ├── step12_evaluate.py     # Evaluate on validation set
│   ├── step13_tune.py         # Tune XGBoost hyperparameters
│   ├── step14_test.py         # Final evaluation on test set
│   ├── step15_plots.py        # Result plots (predicted vs actual, residuals, importance)
│   ├── step16_artifacts.py    # Save and document final artifacts
│   └── predict.py             # Standalone inference script
├── main.py                    # Full pipeline entry point
├── Lab01.ipynb
└── task_02.ipynb
```

---

## Raw Data Format

Each CSV in `data/raw/` represents sensor readings from a fish tank over time.

| Column           | Description                              |
|------------------|------------------------------------------|
| `created_at`     | Timestamp (may include timezone offset)  |
| `entry_id`       | Row identifier                           |
| `TEMPERATURE`    | Water temperature (°C)                   |
| `TURBIDITY`      | Water turbidity (NTU)                    |
| `DISSOLVED OXYGEN` | Dissolved oxygen (mg/L)               |
| `pH`             | Water pH                                 |
| `AMMONIA`        | Ammonia concentration (mg/L)             |
| `NITRATE`        | Nitrate concentration (mg/L)             |
| `Population`     | Fish count in tank                       |
| `Length`         | Fish length (cm)                         |
| `Weight`         | Fish weight (g)                          |

> **Note:** `feed2.csv` contains a column typo (`Lenght`) which is automatically corrected in Step 3.

---

## How Files Are Merged

All CSV files in `data/raw/` are discovered automatically (sorted alphabetically). Each file gets a `source_file` column added before they are concatenated into a single dataframe. This is done in `src/step02_merge.py`.

---

## Preprocessing Steps

| Step | Script                  | What it does                                              |
|------|-------------------------|-----------------------------------------------------------|
| 1    | `step01_inspect.py`     | Lists raw files, checks for missing numbered files        |
| 2    | `step02_merge.py`       | Loads all CSVs, adds `source_file`, concatenates          |
| 3    | `step03_standardize.py` | Strips whitespace, renames typo columns (`Lenght→Length`) |
| 4    | `step04_datetime.py`    | Parses `created_at` (handles mixed timezones), sorts      |
| 5    | `step05_clean.py`       | Removes exact duplicates, reports missing values          |
| 6    | `step06_eda.py`         | Generates distribution plots, time-series plots, summary  |

---

## How the Target Is Created

**Chosen target: `next_weight`**

In Step 7, the `Weight` column is shifted forward by one row to create `next_weight` — the weight measurement at the *next* time step. This is the value the model is trained to predict.

**Why `next_weight` over `weight_gain`:**
- `next_weight` is directly interpretable as an absolute fish weight (grams), which is actionable for fish farmers planning harvest or feeding schedules.
- `weight_gain` (the difference) is small and noisy between consecutive readings, making it harder for models to learn a clear signal.
- Predicting the next absolute weight is standard practice in fish growth modeling.

After aggregation in Step 8, `next_weight` is recomputed as the next window's mean weight.

---

## Feature Engineering (Step 9)

Temporal features are built from the 6-hour aggregated windows:

- **Lag features** (lags 1, 2, 3 windows): `<sensor>_mean_lag1`, `_lag2`, `_lag3`
- **Rolling features** (windows of 3, 6): mean, min, max, std over the prior N windows
- **Calendar features**: `hour`, `day_of_week`, `month`

Rows with insufficient history (NaN lags/rolling) are dropped before modeling.

---

## Why Chronological Validation

Data is split **chronologically** (not randomly):
- **Train**: first 70% of rows by time
- **Val**: next 15%
- **Test**: final 15%

This prevents **data leakage** — future sensor readings must never inform predictions about the past. Random splits would allow a model to "see the future" during training, resulting in falsely optimistic metrics.

---

## Models Trained

| Model              | Notes                             |
|--------------------|-----------------------------------|
| Linear Regression  | Baseline with StandardScaler      |
| Random Forest      | 100 trees, no scaling needed      |
| XGBoost            | Tuned via grid search on val set  |

The best model (lowest validation RMSE after tuning) is retrained on train+val combined and evaluated on the held-out test set.

---

## How to Run

### Full Pipeline

```bash
cd /home/runner/work/DeepLearningLab01/DeepLearningLab01
python main.py
```

This runs all 16 steps end-to-end. All outputs are written to `data/`, `models/`, and `reports/`.

### Run Individual Steps

```bash
python src/step01_inspect.py
python src/step06_eda.py
python src/step11_train.py
# ... etc.
```

### Run Inference on New Data

```bash
python src/predict.py --input data/processed/model_ready.csv --output predictions.csv
```

The input CSV must contain the same feature columns used during training (stored in `models/feature_list.pkl`).

---

## Configuration

All paths, hyperparameters, and settings are centralized in `src/config.py`:

```python
AGGREGATION_WINDOW = "6h"   # Time window for sensor aggregation
LAG_PERIODS = [1, 2, 3]     # Lag steps for temporal features
ROLLING_WINDOWS = [3, 6]    # Rolling window sizes
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
```

---

## Outputs

| Output                                  | Description                              |
|-----------------------------------------|------------------------------------------|
| `reports/logs/file_summary.txt`         | Raw file inventory                       |
| `reports/logs/missing_value_report.txt` | Missing value counts per column          |
| `reports/metrics/eda_summary.txt`       | Descriptive stats and value frequencies  |
| `reports/metrics/target_decision.txt`   | Justification for chosen target          |
| `reports/metrics/split_summary.txt`     | Train/val/test split statistics          |
| `reports/metrics/validation_results.csv`| MAE, RMSE, R² for all models on val set  |
| `reports/metrics/best_params.json`      | Best XGBoost hyperparameters             |
| `reports/metrics/test_results.json`     | Final test MAE, RMSE, R²                 |
| `reports/figures/*.png`                 | Distribution, time-series, importance plots |
| `models/best_model.pkl`                 | Best trained model                       |
| `models/artifact_manifest.json`         | Paths to all saved artifacts             |

---

## Assumptions and Limitations

- **Sensor data quality**: The pipeline assumes sensor readings are roughly continuous. Large gaps between files are handled by the chronological sort, but no gap interpolation is performed.
- **Target definition**: `next_weight` is defined at the 6-hour aggregated window level. It reflects the average weight of sampled fish in the next window — not an individual fish.
- **Single tank assumption**: All CSV files are assumed to come from comparable fish tanks with compatible sensor semantics. Cross-tank normalization is not performed.
- **No outlier removal**: Step 5 only removes exact duplicates. Physiologically implausible sensor values are not filtered. For production use, domain-specific range checks should be added.
- **XGBoost tuning scope**: The hyperparameter grid is intentionally small (16 configurations) for demonstration. A full production search would use a larger grid or Bayesian optimization.
- **Feature leakage guard**: Rolling features are computed with `shift(1)` to ensure no same-window information leaks into features.
