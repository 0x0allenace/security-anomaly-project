import pandas as pd
import numpy as np

# Load dataset


def load_data(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# Time-based features
def add_time_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday

    # Night activity flag (potential anomaly)
    df["is_night"] = df["hour"].apply(lambda x: 1 if x < 6 else 0)

    return df


# Behavioral aggregation per user
def add_user_behavior_features(df):

    # Sort for rolling operations
    df = df.sort_values(by=["user_id", "timestamp"])

    # Login frequency (rolling count)
    df["login_attempt"] = df["action"].apply(
        lambda x: 1 if x == "login" else 0)

    df["login_freq_5"] = df.groupby("user_id")["login_attempt"]\
        .rolling(window=5, min_periods=1)\
        .sum().reset_index(level=0, drop=True)

    # Failed login rolling
    df["failed_attempts_rolling"] = df.groupby("user_id")["failed_attempts"]\
        .rolling(window=5, min_periods=1)\
        .mean().reset_index(level=0, drop=True)

    return df


# Statistical features
def add_statistical_features(df):

    # Z-score for session duration
    df["session_zscore"] = (df["session_duration_min"] -
                            df["session_duration_min"].mean()) / df["session_duration_min"].std()

    # Z-score for failed attempts
    df["failed_zscore"] = (
        df["failed_attempts"] - df["failed_attempts"].mean()) / df["failed_attempts"].std()

    return df


# User baseline behavior
def add_user_baseline(df):

    user_avg_session = df.groupby(
        "user_id")["session_duration_min"].transform("mean")
    df["session_deviation"] = df["session_duration_min"] - user_avg_session

    return df

# Risk score


def compute_risk_score(df):

    df["risk_score"] = (
        (df["failed_attempts"] * 0.4) +
        (df["is_night"] * 0.2) +
        (abs(df["session_zscore"]) * 0.2) +
        (df["login_freq_5"] * 0.2)
    )

    return df


def add_moving_averages(df, window=5):
    """
    Add moving average features for temporal smoothing (per user).

    Parameters:
        df (pd.DataFrame): Input dataframe
        window (int): rolling window size

    Returns:
        df (pd.DataFrame): dataframe with new features
    """

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by user + time
    df = df.sort_values(by=['user_id', 'timestamp'])

    # Moving averages (per user)
    df[f'risk_score_ma_{window}'] = (
        df.groupby('user_id')['risk_score']
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df[f'login_freq_ma_{window}'] = (
        df.groupby('user_id')['login_freq_5']
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Deviation from moving average
    df[f'risk_score_dev_{window}'] = (
        df['risk_score'] - df[f'risk_score_ma_{window}']
    )

    return df


def add_time_window_aggregation(df, window='5min'):
    """
    Aggregate user activity over time windows.

    Parameters:
        df (pd.DataFrame): input dataframe
        window (str): time window (e.g., '5T' = 5 minutes)

    Returns:
        df_agg (pd.DataFrame): aggregated dataframe
    """

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort values
    df = df.sort_values(by='timestamp')

    # Set index for resampling
    df = df.set_index('timestamp')

    # Group by user + time window
    df_agg = (
        df.groupby('user_id')
          .resample(window)
          .agg({
              'login_freq_5': 'sum',
              'failed_attempts': 'sum',
              'risk_score': 'mean'
          })
        .reset_index()
    )

    # Rename columns for clarity
    df_agg.rename(columns={
        'login_freq_5': f'login_count_{window}',
        'failed_attempts': f'failed_attempts_{window}',
        'risk_score': f'risk_score_mean_{window}'
    }, inplace=True)

    return df_agg

# Main pipeline


def run_feature_engineering(input_path, output_path):

    df = load_data(input_path)

    df = add_time_features(df)
    df = add_user_behavior_features(df)
    df = add_statistical_features(df)
    df = add_user_baseline(df)
    df = compute_risk_score(df)

    df.to_csv(output_path, index=False)

    print("Feature engineering completed.")
    print(f"Saved to: {output_path}")
    print(df.head())


if __name__ == "__main__":
    run_feature_engineering(
        input_path="data/raw/logs_with_attacks.csv",
        output_path="data/processed/processed_logs.csv"
    )
