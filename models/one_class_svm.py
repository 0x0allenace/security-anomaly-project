import pandas as pd
from sklearn.svm import OneClassSVM


def run_one_class_svm(
    df,
    feature_cols=None,
    kernel="rbf",
    gamma="scale",
    nu=0.05
):
    """
    Apply One-Class SVM for anomaly detection.

    Parameters:
        df (pd.DataFrame): Input dataframe with engineered features
        feature_cols (list): List of feature columns to use
        kernel (str): Kernel type
        gamma (str or float): Kernel coefficient
        nu (float): Upper bound on anomaly fraction and lower bound on support vectors

    Returns:
        df_model (pd.DataFrame): Dataframe with SVM results added
    """

    if feature_cols is None:
        feature_cols = [
            "risk_score",
            "session_duration_min",
            "login_freq_5",
            "failed_attempts_rolling",
            "session_zscore",
            "failed_zscore"
        ]

    df_model = df.copy()
    df_model = df_model.dropna(subset=feature_cols)

    X = df_model[feature_cols].copy()
    X = X.fillna(0)

    ocsvm = OneClassSVM(
        kernel=kernel,
        gamma=gamma,
        nu=nu
    )

    ocsvm.fit(X)

    svm_labels = ocsvm.predict(X)

    # Convert output: -1 = anomaly, 1 = normal
    df_model["svm_label"] = svm_labels
    df_model["svm_anomaly"] = df_model["svm_label"].apply(
        lambda x: 1 if x == -1 else 0)

    # Decision score
    df_model["svm_score"] = ocsvm.decision_function(X)

    return df_model


def save_svm_results(df, output_path="data/processed/processed_logs_with_svm.csv"):
    """
    Save One-Class SVM results to CSV.

    Parameters:
        df (pd.DataFrame): Dataframe with SVM results
        output_path (str): Output file path
    """
    df.to_csv(output_path, index=False)
    print(f"One-Class SVM results saved to {output_path}")


def print_svm_summary(df):
    """
    Print a simple anomaly detection summary.
    """
    print("\nOne-Class SVM Summary")
    print(df["svm_anomaly"].value_counts())

    if "is_attack" in df.columns:
        print("\nGround Truth vs One-Class SVM")
        print(pd.crosstab(df["is_attack"], df["svm_anomaly"]))


def main():
    input_path = "data/processed/processed_logs_with_stats.csv"
    output_path = "data/processed/processed_logs_with_svm.csv"

    df = pd.read_csv(input_path)

    df_svm = run_one_class_svm(df)

    save_svm_results(df_svm, output_path=output_path)
    print_svm_summary(df_svm)

    print("\nPreview:")
    print(df_svm.head())


if __name__ == "__main__":
    main()
