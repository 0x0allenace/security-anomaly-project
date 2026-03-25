import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# This script performs statistical analysis on the processed log data to identify potential anomalies based on session duration, login frequency, and risk scores. It computes z-scores for these features and applies both z-score and IQR methods to flag anomalous entries. The results are printed to the console and the processed dataset with statistical features is saved for further use in model training and evaluation.


def compute_zscore(df, column):
    df = df.copy()
    mean = df[column].mean()
    std = df[column].std()

    if std == 0 or pd.isna(std):
        df[f"{column}_zscore"] = 0
    else:
        df[f"{column}_zscore"] = (df[column] - mean) / std

    return df

# Detect anomalies using z-score method (threshold of 3 standard deviations) and IQR method for a given column


def detect_anomalies_zscore(df, column, threshold=3):
    return df[np.abs(df[f"{column}_zscore"]) > threshold]

# Detect anomalies using IQR method for a given column


def detect_anomalies_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return df[(df[column] < lower) | (df[column] > upper)]

# Main function to run statistical analysis on the processed logs and print results


def run_statistical_analysis(df):
    df = df.copy()

    columns = [
        "session_duration_min",
        "login_freq_5",
        "risk_score",
    ]

    results = {}

    for col in columns:
        df = compute_zscore(df, col)

        z_anomalies = detect_anomalies_zscore(df, col)
        iqr_anomalies = detect_anomalies_iqr(df, col)

        results[col] = {
            "zscore_anomalies": len(z_anomalies),
            "iqr_anomalies": len(iqr_anomalies),
        }

    return df, results

# Seasonal decomposition for a given column to analyze trends and seasonality in the data


def run_seasonal_decomposition(df, column, period=12, model="additive"):
    """
    Perform seasonal decomposition on a time series column.

    Parameters:
        df (pd.DataFrame): input dataframe
        column (str): column to decompose
        period (int): seasonal period
        model (str): 'additive' or 'multiplicative'

    Returns:
        decomposition: statsmodels decomposition result
    """

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Build time series
    ts = df.set_index("timestamp")[column]

    # If duplicate timestamps exist, aggregate first
    ts = ts.groupby(level=0).mean()

    # Fill missing timestamps by forward fill after resampling
    ts = ts.resample("5min").mean().ffill()

    decomposition = seasonal_decompose(ts, model=model, period=period)

    return decomposition


def plot_seasonal_decomposition(decomposition, column):
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.suptitle(f"Seasonal Decomposition of {column}", y=1.02)
    plt.show()

# Function to print the results of the statistical analysis in a readable format


def print_results(results):
    print("\nStatistical Anomaly Detection Results:")
    for col, res in results.items():
        print(f"\n{col}:")
        print(f"  Z-score anomalies: {res['zscore_anomalies']}")
        print(f"  IQR anomalies: {res['iqr_anomalies']}")

# Main function to execute the statistical analysis pipeline, read the processed logs, compute anomalies, and save the results with statistical features for further use in model training and evaluation.


def main():
    input_path = "data/processed/processed_logs.csv"
    output_path = "data/processed/processed_logs_with_stats.csv"

    df = pd.read_csv(input_path)
    df, results = run_statistical_analysis(df)

    df.to_csv(output_path, index=False)

    print(f"Processed statistical analysis saved to: {output_path}")
    print_results(results)
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()
