import pandas as pd

from evaluation.metrics import (
    compute_classification_metrics,
    metrics_dict_to_dataframe,
)


def evaluate_model(df, true_col, pred_col, model_name):
    """
    Evaluate a single anomaly detection model.

    Parameters:
        df (pd.DataFrame): Model output dataframe
        true_col (str): Ground truth column
        pred_col (str): Predicted anomaly column
        model_name (str): Name of the model

    Returns:
        dict: Evaluation metrics
    """

    metrics = compute_classification_metrics(
        df[true_col],
        df[pred_col]
    )

    metrics["model"] = model_name

    return metrics


def evaluate_all_models():
    """
    Load all saved model outputs and evaluate them side by side.

    Models evaluated:
        1. Isolation Forest
        2. LOF
        3. One-Class SVM
        4. Autoencoder
        5. Gaussian Mixture Model (GMM)

    Returns:
        pd.DataFrame: Comparative evaluation results
    """

    # Load model outputs
    df_if = pd.read_csv(
        "data/processed/processed_logs_with_iforest.csv"
    )

    df_lof = pd.read_csv(
        "data/processed/processed_logs_with_lof.csv"
    )

    df_svm = pd.read_csv(
        "data/processed/processed_logs_with_svm.csv"
    )

    df_auto = pd.read_csv(
        "data/processed/processed_logs_with_autoencoder.csv"
    )

    df_gmm = pd.read_csv(
        "data/processed/processed_logs_with_gmm.csv"
    )

    # Compute evaluation metrics for each model
    results = {

        "Isolation Forest": compute_classification_metrics(
            df_if["is_attack"],
            df_if["iforest_anomaly"]
        ),

        "LOF": compute_classification_metrics(
            df_lof["is_attack"],
            df_lof["lof_anomaly"]
        ),

        "One-Class SVM": compute_classification_metrics(
            df_svm["is_attack"],
            df_svm["svm_anomaly"]
        ),

        "Autoencoder": compute_classification_metrics(
            df_auto["is_attack"],
            df_auto["autoencoder_anomaly"]
        ),

        "GMM": compute_classification_metrics(
            df_gmm["is_attack"],
            df_gmm["gmm_anomaly"]
        ),
    }

    # Convert metrics dictionary to dataframe
    results_df = metrics_dict_to_dataframe(results)

    # Maintain consistent column order
    ordered_cols = [
        "model",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1_score",
        "accuracy",
    ]

    results_df = results_df[ordered_cols]

    return results_df


def main():
    """
    Run model evaluation and save comparative results.
    """

    results_df = evaluate_all_models()

    print("\nModel Comparison Results:\n")
    print(results_df.round(4))

    # Save results
    output_path = (
        "data/processed/model_comparison_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved evaluation results to: {output_path}"
    )


if __name__ == "__main__":
    main()
