"""
Step 6: Exploratory Data Analysis.
- Analyze Length and Weight target columns
- Create distribution and time-series plots
- Save EDA summary
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.config import CLEANED_DATA_PATH, FIGURES_DIR, EDA_SUMMARY_PATH, SENSOR_COLUMNS


def run_eda():
    """Run exploratory data analysis."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = pd.read_csv(CLEANED_DATA_PATH, parse_dates=["created_at"])
    print(f"Loaded cleaned data: {df.shape}")

    unique_lengths = df["Length"].nunique()
    unique_weights = df["Weight"].nunique()
    weight_freq = df["Weight"].value_counts().head(10)
    length_freq = df["Length"].value_counts().head(10)

    print(f"Unique Length values: {unique_lengths}")
    print(f"Unique Weight values: {unique_weights}")

    # Weight distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["Weight"].dropna(), bins=30, color="steelblue", edgecolor="white")
    ax.set_title("Weight Distribution")
    ax.set_xlabel("Weight (g)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "weight_distribution.png"))
    plt.close()

    # Length distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["Length"].dropna(), bins=30, color="coral", edgecolor="white")
    ax.set_title("Length Distribution")
    ax.set_xlabel("Length (cm)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "length_distribution.png"))
    plt.close()

    # Weight over time
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["created_at"], df["Weight"], alpha=0.6, color="steelblue", linewidth=0.8)
    ax.set_title("Weight Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight (g)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "weight_over_time.png"))
    plt.close()

    # Length over time
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["created_at"], df["Length"], alpha=0.6, color="coral", linewidth=0.8)
    ax.set_title("Length Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Length (cm)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "length_over_time.png"))
    plt.close()

    # Sensor trends over time
    available_sensors = [c for c in SENSOR_COLUMNS if c in df.columns]
    fig, axes = plt.subplots(len(available_sensors), 1, figsize=(14, 3 * len(available_sensors)), sharex=True)
    if len(available_sensors) == 1:
        axes = [axes]
    for ax, col in zip(axes, available_sensors):
        ax.plot(df["created_at"], df[col], alpha=0.6, linewidth=0.8)
        ax.set_ylabel(col, fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date")
    axes[0].set_title("Sensor Trends Over Time")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "sensor_trends_over_time.png"))
    plt.close()

    print(f"Plots saved to {FIGURES_DIR}")

    os.makedirs(os.path.dirname(EDA_SUMMARY_PATH), exist_ok=True)
    with open(EDA_SUMMARY_PATH, "w") as f:
        f.write("EDA SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total rows: {len(df)}\n")
        f.write(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}\n\n")
        f.write(f"Unique Length values: {unique_lengths}\n")
        f.write(f"Unique Weight values: {unique_weights}\n\n")
        f.write("Top 10 Weight frequencies:\n")
        f.write(weight_freq.to_string())
        f.write("\n\nTop 10 Length frequencies:\n")
        f.write(length_freq.to_string())
        f.write("\n\nDescriptive Statistics:\n")
        numeric_cols = df.select_dtypes(include="number").columns
        f.write(df[numeric_cols].describe().to_string())

    print(f"EDA summary saved to {EDA_SUMMARY_PATH}")
    return df


if __name__ == "__main__":
    run_eda()
