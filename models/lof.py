import pandas as pd
from sklearn.neighbors import LocalOutlierFactor


def run_lof(
    df,
    feature_cols=None,
    n_neighbors=20,
    contamination=0.05
):
    """
    Apply Local Outlier Factor (LOF) for anomaly detection.

    Parameters:
        df (pd.DataFrame): Input dataframe with engineered features
        feature_cols (list): List of feature columns to use
        n_neighbors (int): Number of neighbors for LOF
        contamination (float): Expected proportion of anomalies

    Returns:
        df (pd.DataFrame): Dataframe with LOF results added
    """

    # Default feature set (aligned with your project)
    if feature_cols is None:
        feature_cols = [
            "risk_score",
            "session_duration_min",
            "login_freq_5",
            "failed_attempts_rolling",
            "session_zscore",
            "failed_zscore"
        ]

    # Ensure features exist
    df_model = df.copy()
    df_model = df_model.dropna(subset=feature_cols)

    X = df_model[feature_cols]

    # Initialize LOF model
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )

    # Fit and predict
    lof_labels = lof.fit_predict(X)

    # Convert LOF output:
    # -1 = anomaly, 1 = normal
    df_model["lof_label"] = lof_labels
    df_model["lof_anomaly"] = df_model["lof_label"].apply(
        lambda x: 1 if x == -1 else 0)

    # LOF scores (negative_outlier_factor_)
    df_model["lof_score"] = lof.negative_outlier_factor_

    return df_model


def save_lof_results(df, output_path="../data/processed/processed_logs_with_lof.csv"):
    """
    Save LOF results to CSV.

    Parameters:
        df (pd.DataFrame): Dataframe with LOF results
        output_path (str): Output file path
    """

    df.to_csv(output_path, index=False)
    print(f"LOF results saved to {output_path}")
