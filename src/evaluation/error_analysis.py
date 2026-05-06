"""
Error analysis: analyze false positives and false negatives for model improvement.
"""
import pandas as pd
import numpy as np


def error_analysis(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Analyze false positives and false negatives.

    Args:
        df: Enriched transaction DataFrame aligned with y_true/y_pred.
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        dict with 'false_positives' and 'false_negatives' DataFrames
        and summary statistics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    false_positives = df[fp_mask].copy()
    false_negatives = df[fn_mask].copy()

    summary = {
        "total_predictions": len(y_pred),
        "false_positive_count": int(fp_mask.sum()),
        "false_negative_count": int(fn_mask.sum()),
        "false_positive_rate": float(fp_mask.sum() / max((y_true == 0).sum(), 1)),
        "false_negative_rate": float(fn_mask.sum() / max((y_true == 1).sum(), 1)),
    }

    # Summarize FP characteristics
    if not false_positives.empty and "amount" in false_positives.columns:
        summary["fp_avg_amount"] = float(false_positives["amount"].mean())
        summary["fp_night_ratio"] = float(false_positives.get("is_night", pd.Series([0])).mean())

    # Summarize FN characteristics
    if not false_negatives.empty and "amount" in false_negatives.columns:
        summary["fn_avg_amount"] = float(false_negatives["amount"].mean())
        summary["fn_night_ratio"] = float(false_negatives.get("is_night", pd.Series([0])).mean())

    return {
        "summary": summary,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
