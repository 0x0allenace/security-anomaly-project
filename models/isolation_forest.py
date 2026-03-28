from sklearn.ensemble import IsolationForest
import pandas as pd


def run_isolation_forest(df, feature_cols, contamination=0.05, random_state=42):
    """
    Apply Isolation Forest for anomaly detection.

    Parameters:
        df (pd.DataFrame): input dataframe
        feature_cols (list): list of feature columns to use
        contamination (float): expected proportion of anomalies
        random_state (int): reproducibility seed

    Returns:
        df (pd.DataFrame): dataframe with anomaly labels and scores
        model (IsolationForest): trained model
    """

    df = df.copy()

    # Select features
    X = df[feature_cols].copy()

    # Fill missing values if any
    X = X.fillna(0)

    # Initialize model
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state
    )

    # Fit model
    model.fit(X)

    # Predict anomalies (-1 = anomaly, 1 = normal)
    df["iforest_label"] = model.predict(X)

    # Convert to 0 (normal) / 1 (anomaly)
    df["iforest_anomaly"] = df["iforest_label"].apply(
        lambda x: 1 if x == -1 else 0)

    # Anomaly score (lower = more abnormal)
    df["iforest_score"] = model.decision_function(X)

    return df, model


def get_default_iforest_features():
    """
    Return the default feature set for Isolation Forest.
    """
    return [
        "risk_score",
        "session_duration_min",
        "login_freq_5",
        "failed_attempts_rolling",
        "session_zscore",
        "failed_zscore"
    ]


def print_iforest_summary(df):
    """
    Print a simple anomaly detection summary.
    """
    print("\nIsolation Forest Summary")
    print(df["iforest_anomaly"].value_counts())

    if "is_attack" in df.columns:
        print("\nGround Truth vs Isolation Forest")
        print(pd.crosstab(df["is_attack"], df["iforest_anomaly"]))


def main():
    input_path = "data/processed/processed_logs_with_stats.csv"
    output_path = "data/processed/processed_logs_with_iforest.csv"

    df = pd.read_csv(input_path)

    feature_cols = get_default_iforest_features()

    df, model = run_isolation_forest(
        df=df,
        feature_cols=feature_cols,
        contamination=0.05,
        random_state=42
    )

    df.to_csv(output_path, index=False)

    print(f"Isolation Forest results saved to: {output_path}")
    print_iforest_summary(df)
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()
