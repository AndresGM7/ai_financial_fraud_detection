"""
evaluation/metrics.py
─────────────────────
Comprehensive evaluation metrics for fraud detection systems.

Metrics implemented
───────────────────
  • Precision, Recall, F1, F-beta
  • PR-AUC (preferred over ROC-AUC for imbalanced data)
  • ROC-AUC (reported for completeness, not primary)
  • Cost-matrix weighted score
  • Confusion matrix with rates
  • Optimal threshold via F-beta maximisation

Interview talking points
────────────────────────
  "In fraud detection, precision-recall tradeoffs are everything:
   - High recall = catch more fraud, but more false positives → alert fatigue
   - High precision = fewer false positives, but more missed fraud

  I use F-beta with beta=2 as the PRIMARY metric because catching fraud
  (recall) is twice as important as not annoying users (precision) in an
  elder-protection context. Carefull's mission is safety-first.

  PR-AUC is preferred over ROC-AUC for imbalanced data because ROC-AUC
  is inflated by the large number of true negatives — it makes a model
  that flags 1% of transactions correctly look great because it gets
  99% of negatives right by default. PR-AUC doesn't have this problem.

  Cost matrix: a missed fraud alert (false negative) for a $15,000 wire
  costs ~$15,000. A false positive costs ~$5 in customer service time.
  Weighting by cost gives a business-relevant scalar metric that aligns
  ML optimisation with actual financial impact."
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import get_logger

log = get_logger(__name__)


# ── Core evaluation function ──────────────────────────────────────────────────

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    beta: float = 2.0,
    cost_fn: float = 15000.0,
    cost_fp: float = 5.0,
) -> dict[str, Any]:
    """
    Compute comprehensive fraud detection metrics.

    Parameters
    ----------
    y_true  : ground truth labels (0/1)
    y_pred  : binary predictions (0/1)
    y_score : probability scores for PR-AUC / ROC-AUC (optional)
    beta    : F-beta weight; beta=2 means recall weighted 2× over precision
    cost_fn : cost of one false negative (missed fraud) in USD
    cost_fp : cost of one false positive (false alert) in USD

    Returns
    -------
    dict with all computed metrics
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f_beta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

    # Cost-weighted score (lower is better — total financial loss)
    total_cost = float(fn * cost_fn + fp * cost_fp)
    # Normalised to [0,1]: 0 = no cost, 1 = worst case
    max_cost = float(len(y_true) * max(cost_fn, cost_fp))
    cost_score = 1.0 - (total_cost / max(max_cost, 1.0))

    metrics: dict[str, Any] = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        f"f{beta}": round(f_beta, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
        "false_negative_rate": round(fn / max(fn + tp, 1), 4),
        "fraud_rate_true": round(y_true.mean(), 4),
        "fraud_rate_pred": round(y_pred.mean(), 4),
        "cost_fn_per_event_usd": cost_fn,
        "cost_fp_per_event_usd": cost_fp,
        "total_cost_usd": round(total_cost, 2),
        "cost_weighted_score": round(cost_score, 4),
        "confusion_matrix": cm.tolist(),
    }

    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        if len(np.unique(y_true)) > 1:
            metrics["pr_auc"] = round(average_precision_score(y_true, y_score), 4)
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_score), 4)
        else:
            metrics["pr_auc"] = None
            metrics["roc_auc"] = None

    log.info("evaluation_complete", **{k: v for k, v in metrics.items()
                                       if k != "confusion_matrix"})
    return metrics


# ── Threshold optimisation ────────────────────────────────────────────────────

def optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    beta: float = 2.0,
) -> tuple[float, dict[str, Any]]:
    """
    Find the decision threshold that maximises F-beta score.

    Interview talking point:
      "The default 0.5 threshold is almost never optimal for imbalanced
       fraud data. I sweep thresholds using the precision-recall curve and
       pick the one that maximises F-beta. This is equivalent to finding
       the operating point on the PR curve that minimises business cost."
    """
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_score)

    # F-beta at each threshold
    f_scores = (
        (1 + beta**2) * precision_arr * recall_arr /
        (beta**2 * precision_arr + recall_arr + 1e-8)
    )

    best_idx = np.argmax(f_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    summary = {
        "optimal_threshold": round(best_threshold, 4),
        f"f{beta}_at_optimal": round(float(f_scores[best_idx]), 4),
        "precision_at_optimal": round(float(precision_arr[best_idx]), 4),
        "recall_at_optimal": round(float(recall_arr[best_idx]), 4),
    }
    log.info("threshold_optimised", **summary)
    return best_threshold, summary


# ── PR curve data (for plotting) ──────────────────────────────────────────────

def pr_curve_data(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label: str = "model",
) -> pd.DataFrame:
    """Return a DataFrame of (threshold, precision, recall) for plotting."""
    p, r, t = precision_recall_curve(y_true, y_score)
    return pd.DataFrame({
        "threshold": list(t) + [1.0],
        "precision": p,
        "recall": r,
        "model": label,
    })

