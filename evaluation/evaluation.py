import pandas as pd

from evaluation.metrics import (
    compute_classification_metrics,
    metrics_dict_to_dataframe,
)


def evaluate_model(df, true_col, pred_col, model_name):
    """
    Evaluate a single model.

    Parameters:
        df (pd.DataFrame): Model output dataframe
        true_col (str): Ground truth column
        pred_col (str): Prediction column
        model_name (str): Name of model

    Returns:
        dict: evaluation metrics
    """
    metrics = compute_classification_metrics(df[true_col], df[pred_col])
    metrics["model"] = model_name
    return metrics


def evaluate_all_models():
    """
    Load all saved model outputs and evaluate them side by side.
    """
    df_if = pd.read_csv("data/processed/processed_logs_with_iforest.csv")
    df_lof = pd.read_csv("data/processed/processed_logs_with_lof.csv")
    df_svm = pd.read_csv("data/processed/processed_logs_with_svm.csv")
    df_auto = pd.read_csv("data/processed/processed_logs_with_autoencoder.csv")

    results = {
        "Isolation Forest": compute_classification_metrics(
            df_if["is_attack"], df_if["iforest_anomaly"]
        ),
        "LOF": compute_classification_metrics(
            df_lof["is_attack"], df_lof["lof_anomaly"]
        ),
        "One-Class SVM": compute_classification_metrics(
            df_svm["is_attack"], df_svm["svm_anomaly"]
        ),
        "Autoencoder": compute_classification_metrics(
            df_auto["is_attack"], df_auto["autoencoder_anomaly"]
        ),
    }

    results_df = metrics_dict_to_dataframe(results)

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
    results_df = evaluate_all_models()

    print("\nModel Comparison Results:\n")
    print(results_df.round(4))

    output_path = "data/processed/model_comparison_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved evaluation results to: {output_path}")


if __name__ == "__main__":
    main()
