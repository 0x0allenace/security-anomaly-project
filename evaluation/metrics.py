import pandas as pd


def compute_confusion_counts(y_true, y_pred):
    """
    Compute confusion matrix counts for binary anomaly detection.

    Parameters:
        y_true (pd.Series): Ground truth labels (0 = normal, 1 = attack)
        y_pred (pd.Series): Predicted labels (0 = normal, 1 = anomaly)

    Returns:
        dict: TP, TN, FP, FN
    """
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    return {
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def compute_classification_metrics(y_true, y_pred):
    """
    Compute evaluation metrics for binary anomaly detection.

    Parameters:
        y_true (pd.Series): Ground truth labels
        y_pred (pd.Series): Predicted labels

    Returns:
        dict: confusion counts + precision, recall, f1, accuracy
    """
    counts = compute_confusion_counts(y_true, y_pred)

    tp = counts["true_positives"]
    tn = counts["true_negatives"]
    fp = counts["false_positives"]
    fn = counts["false_negatives"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / (tp + tn + fp +
                            fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
    }


def build_confusion_table(y_true, y_pred):
    """
    Return a pandas crosstab confusion matrix.
    """
    return pd.crosstab(
        pd.Series(y_true, name="is_attack"),
        pd.Series(y_pred, name="predicted_anomaly")
    )


def metrics_dict_to_dataframe(metrics_dict):
    """
    Convert a nested metrics dictionary into a dataframe.

    Parameters:
        metrics_dict (dict): {"Model Name": {...metrics...}, ...}

    Returns:
        pd.DataFrame
    """
    df = pd.DataFrame(metrics_dict).T.reset_index()
    df = df.rename(columns={"index": "model"})
    return df
