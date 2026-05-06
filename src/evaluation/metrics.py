"""
Evaluation metrics: precision, recall, F1, confusion matrix, and ROC-AUC.
"""
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray = None) -> dict:
    """
    Compute classification metrics for fraud detection.

    Args:
        y_true: Ground-truth binary labels (0=legit, 1=fraud).
        y_pred: Predicted binary labels.
        y_score: Predicted probabilities for the positive class (optional).

    Returns:
        Dictionary of metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    result = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "support_positive": int(y_true.sum()),
        "support_total": int(len(y_true)),
    }

    if y_score is not None:
        y_score = np.asarray(y_score)
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_score))
            result["avg_precision"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            result["roc_auc"] = None
            result["avg_precision"] = None

    return result
