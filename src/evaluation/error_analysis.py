"""
evaluation/error_analysis.py
────────────────────────────
Systematic False Positive / False Negative analysis.

Analysis cuts
─────────────
  • By fraud type         (which patterns does the model miss?)
  • By merchant category  (which MCCs have high FP rate?)
  • By time-of-day        (night vs. day performance)
  • By transaction type   (ACH vs WIRE vs ZELLE)
  • By amount band        (small vs. large transactions)
  • SHAP waterfall plots  (why did THIS transaction get flagged?)

Interview talking point
───────────────────────
  "Error analysis is where you earn your salary as a data scientist.
   Aggregate metrics like PR-AUC hide failure modes: a model with
   PR-AUC=0.88 might miss 90% of romance scams while catching all the
   card-not-present fraud. Without sliced error analysis, you'd deploy
   a model that completely fails for the highest-value fraud type.

  I also look at False Positives by merchant: if we're alerting on 30%
  of Western Union transactions because they're flagged as high-risk MCC,
  we need to know — those could be legitimate remittances from immigrant
  families. False positives erode trust and cause customer churn."
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_logger

log = get_logger(__name__)

# Try importing SHAP (optional, graceful degradation)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ── Slice analysis ────────────────────────────────────────────────────────────

def slice_metrics(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    slice_col: str,
    label_col: str = "is_fraud",
) -> pd.DataFrame:
    """
    Compute precision, recall, FPR per slice of a categorical column.

    Usage
    -----
    >>> results = slice_metrics(df, y_pred, slice_col="fraud_type")
    >>> results = slice_metrics(df, y_pred, slice_col="tx_type")
    """
    df = df.copy()
    df["_y_pred"] = y_pred
    df["_y_true"] = df[label_col].fillna(0).astype(int)

    # Convert Categorical to string before fillna to avoid Categorical type error
    col_series = df[slice_col].astype(str).fillna("unknown")

    rows = []
    for val in col_series.unique():
        mask = col_series == val
        sub = df[mask]
        y_t = sub["_y_true"].values
        y_p = sub["_y_pred"].values

        tp = int(((y_t == 1) & (y_p == 1)).sum())
        fp = int(((y_t == 0) & (y_p == 1)).sum())
        fn = int(((y_t == 1) & (y_p == 0)).sum())
        tn = int(((y_t == 0) & (y_p == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        rows.append({
            slice_col: val,
            "n_samples": len(sub),
            "n_fraud": int(y_t.sum()),
            "fraud_rate": round(y_t.mean(), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "fpr": round(fpr, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    return pd.DataFrame(rows).sort_values("n_fraud", ascending=False)


def full_error_analysis(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    label_col: str = "is_fraud",
) -> dict[str, pd.DataFrame]:
    """
    Run slice analysis across multiple dimensions.

    Returns a dict of DataFrames: {dimension_name: slice_metrics_df}
    """
    results: dict[str, pd.DataFrame] = {}
    dimensions = {
        "fraud_type": "fraud_type",
        "tx_type": "tx_type",
        "is_night": "is_night",
        "is_weekend": "is_weekend",
        "mcc_high_risk": "mcc_high_risk",
        "is_new_counterparty": "is_new_counterparty",
    }
    for name, col in dimensions.items():
        if col in df.columns:
            results[name] = slice_metrics(df, y_pred, slice_col=col, label_col=label_col)

    # Amount band analysis
    if "amount" in df.columns:
        df_copy = df.copy()
        df_copy["amount_band"] = pd.cut(
            df_copy["amount"],
            bins=[0, 100, 500, 1000, 5000, 10000, float("inf")],
            labels=["<$100", "$100–500", "$500–1k", "$1k–5k", "$5k–10k", ">$10k"],
        )
        results["amount_band"] = slice_metrics(df_copy, y_pred, "amount_band", label_col)

    log.info("error_analysis_complete", dimensions=list(results.keys()))
    return results


# ── FP / FN sample inspection ─────────────────────────────────────────────────

def get_false_positives(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = 20,
) -> pd.DataFrame:
    """Return top-n false positive transactions for manual review."""
    mask = (y_true == 0) & (y_pred == 1)
    return df[mask].head(n)


def get_false_negatives(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = 20,
) -> pd.DataFrame:
    """Return top-n false negative transactions (missed fraud) for manual review."""
    mask = (y_true == 1) & (y_pred == 0)
    return df[mask].head(n)


# ── SHAP explanations ──────────────────────────────────────────────────────────

def shap_summary(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    max_display: int = 20,
) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values per feature (global importance).

    Interview talking point:
      "SHAP (SHapley Additive exPlanations) gives each feature a contribution
       to the model's prediction for each transaction. For a specific fraud
       alert, I can show the compliance team: 'Z-score contributed +0.32 to
       the risk score, new counterparty +0.28, night hour +0.15.' This is
       the difference between a black-box alert and an auditable decision."
    """
    if not HAS_SHAP:
        log.warning("shap_not_installed")
        return pd.DataFrame()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class

        mean_abs = np.abs(shap_values).mean(axis=0)
        return (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(max_display)
        )
    except Exception as exc:
        log.warning("shap_failed", error=str(exc))
        return pd.DataFrame()


def shap_single_transaction(
    model: Any,
    x: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    SHAP waterfall explanation for a single transaction.
    Used in the API response for human-readable fraud explanations.
    """
    if not HAS_SHAP:
        return pd.DataFrame()
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(x.reshape(1, -1))
        if isinstance(sv, list):
            sv = sv[1]
        return (
            pd.DataFrame({
                "feature": feature_names,
                "shap_value": sv[0],
                "feature_value": x,
            })
            .sort_values("shap_value", key=abs, ascending=False)
        )
    except Exception as exc:
        log.warning("shap_single_failed", error=str(exc))
        return pd.DataFrame()