import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def get_default_gmm_features():
    """
    Return the default feature set used by the GMM model.

    These features are aligned with the feature set used
    by the other anomaly detection models in the project.
    """
    return [
        "risk_score",
        "session_duration_min",
        "login_freq_5",
        "failed_attempts_rolling",
        "session_zscore",
        "failed_zscore"
    ]


def prepare_gmm_data(df, feature_cols):
    """
    Prepare feature data for GMM.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        feature_cols (list): Features used by GMM.

    Returns:
        df_model (pd.DataFrame): Clean dataframe.
        X_scaled (np.ndarray): Standardized feature matrix.
        scaler (StandardScaler): Fitted scaler.
    """

    df_model = df.copy()

    # Remove rows with missing required features
    df_model = df_model.dropna(subset=feature_cols).copy()

    X = df_model[feature_cols].copy()

    # Replace any remaining missing values
    X = X.fillna(0)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df_model, X_scaled, scaler


def run_gmm(
    df,
    feature_cols=None,
    n_components=2,
    covariance_type="full",
    contamination=0.05,
    random_state=42
):
    """
    Apply Gaussian Mixture Model (GMM) for anomaly detection.

    GMM estimates the probability density of observations.
    Observations with low probability under the fitted mixture
    model are treated as potential anomalies.

    Parameters:
        df (pd.DataFrame): Input dataframe with engineered features.
        feature_cols (list): Features used for modeling.
        n_components (int): Number of Gaussian mixture components.
        covariance_type (str): Covariance structure used by GMM.
        contamination (float): Expected proportion of anomalies.
        random_state (int): Random seed for reproducibility.

    Returns:
        df_model (pd.DataFrame): Dataframe with GMM results.
        model (GaussianMixture): Fitted GMM model.
        scaler (StandardScaler): Fitted feature scaler.
        threshold (float): Anomaly score threshold.
    """

    if feature_cols is None:
        feature_cols = get_default_gmm_features()

    # Prepare data
    df_model, X_scaled, scaler = prepare_gmm_data(
        df,
        feature_cols
    )

    # Initialize GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state
    )

    # Fit model
    gmm.fit(X_scaled)

    # Log probability density
    log_likelihood = gmm.score_samples(X_scaled)

    # Convert to anomaly score:
    # Higher value = more anomalous
    anomaly_score = -log_likelihood

    # Determine threshold using contamination
    threshold = np.quantile(
        anomaly_score,
        1 - contamination
    )

    # Generate anomaly labels
    gmm_anomaly = (
        anomaly_score >= threshold
    ).astype(int)

    # Add results
    df_model["gmm_score"] = anomaly_score
    df_model["gmm_log_likelihood"] = log_likelihood
    df_model["gmm_anomaly"] = gmm_anomaly

    # Component assignment
    df_model["gmm_component"] = gmm.predict(X_scaled)

    return df_model, gmm, scaler, threshold


def save_gmm_results(
    df,
    output_path="data/processed/processed_logs_with_gmm.csv"
):
    """
    Save GMM results to CSV.
    """

    df.to_csv(output_path, index=False)

    print(
        f"GMM results saved to {output_path}"
    )


def print_gmm_summary(df):
    """
    Print a simple GMM anomaly detection summary.
    """

    print("\nGaussian Mixture Model Summary")

    print("\nAnomaly Counts:")
    print(
        df["gmm_anomaly"].value_counts()
    )

    if "is_attack" in df.columns:

        print("\nGround Truth vs GMM:")
        print(
            pd.crosstab(
                df["is_attack"],
                df["gmm_anomaly"]
            )
        )

    if "gmm_component" in df.columns:

        print("\nGMM Component Distribution:")
        print(
            df["gmm_component"].value_counts()
        )


def main():

    input_path = (
        "data/processed/processed_logs_with_stats.csv"
    )

    output_path = (
        "data/processed/processed_logs_with_gmm.csv"
    )

    # Load processed dataset
    df = pd.read_csv(input_path)

    # Use project-standard features
    feature_cols = get_default_gmm_features()

    # Run GMM
    df_gmm, model, scaler, threshold = run_gmm(
        df=df,
        feature_cols=feature_cols,
        n_components=2,
        covariance_type="full",
        contamination=0.05,
        random_state=42
    )

    # Save results
    save_gmm_results(
        df_gmm,
        output_path=output_path
    )

    print(
        f"\nGMM anomaly threshold: {threshold:.6f}"
    )

    print_gmm_summary(df_gmm)

    print("\nPreview:")
    print(df_gmm.head())


if __name__ == "__main__":
    main()
