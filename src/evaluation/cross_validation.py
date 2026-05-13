"""
evaluation/cross_validation.py
───────────────────────────────
Time-aware cross-validation for financial transaction data.

Key design decision: TimeSeriesSplit, NOT KFold
────────────────────────────────────────────────
  Standard KFold randomly shuffles data → future transactions appear in
  training set → temporal leakage → optimistic performance estimates.

  TimeSeriesSplit respects temporal order: train on [0..t], test on [t..t+k].
  This mirrors production: model trained on past, evaluated on future.

  Walk-forward validation goes further: re-trains the model after each fold,
  simulating continuous retraining in production.

Interview talking points
────────────────────────
  "This is one of the most common evaluation mistakes I see: using KFold on
   time-series transaction data. Say your model is trained on January–December
   and evaluated on random 20% held-out — that held-out set includes November
   data, which was used in training January folds. The model has 'seen the
   future'. PR-AUC looks great in validation but collapses in production.

  TimeSeriesSplit is the minimum fix. Walk-forward validation is more
  realistic: in production, we retrain weekly on a rolling window, so I
  simulate that exact process in evaluation.

  Bootstrap confidence intervals on metrics give us a proper estimate of
  uncertainty — a point estimate of PR-AUC=0.82 is less useful than
  'PR-AUC = 0.82 ± 0.04 (95% CI)'. This is the statistical rigour that
  separates senior from junior data scientists."
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.evaluation.metrics import evaluate
from src.utils import get_logger

log = get_logger(__name__)


# ── Time-based cross-validation ───────────────────────────────────────────────

def time_series_cv(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.Series,
    n_splits: int = 5,
    beta: float = 2.0,
) -> dict[str, Any]:
    """
    Evaluate a model using TimeSeriesSplit — respects temporal order.

    Parameters
    ----------
    model_factory : callable() → fitted model; called fresh each fold
    X             : feature matrix
    y             : labels
    timestamps    : pd.Series of datetime (used only for logging, not splitting)
    n_splits      : number of folds
    beta          : F-beta weight

    Returns
    -------
    dict with per-fold and aggregated metrics
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model for this fold
        model = model_factory(X_train, y_train)

        # Predict
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
            y_pred = (y_score >= 0.5).astype(int)
        else:
            y_score = None
            y_pred = model.predict(X_test)

        fold_metrics = evaluate(y_test, y_pred, y_score=y_score, beta=beta)
        fold_metrics["fold"] = fold_idx + 1
        fold_metrics["train_size"] = len(train_idx)
        fold_metrics["test_size"] = len(test_idx)
        fold_results.append(fold_metrics)

        log.info(
            "cv_fold_complete",
            fold=fold_idx + 1,
            test_size=len(test_idx),
            pr_auc=fold_metrics.get("pr_auc"),
            recall=fold_metrics["recall"],
        )

    # Aggregate across folds
    numeric_keys = ["precision", "recall", "f1", f"f{beta}", "pr_auc", "roc_auc", "cost_weighted_score"]
    aggregated: dict[str, Any] = {}
    for key in numeric_keys:
        vals = [r[key] for r in fold_results if r.get(key) is not None]
        if vals:
            aggregated[f"{key}_mean"] = round(float(np.mean(vals)), 4)
            aggregated[f"{key}_std"] = round(float(np.std(vals)), 4)

    return {
        "n_splits": n_splits,
        "fold_results": fold_results,
        **aggregated,
    }


# ── Walk-forward validation ───────────────────────────────────────────────────

def walk_forward_validation(
    model_factory: Callable,
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    timestamp_col: str = "timestamp",
    n_folds: int = 5,
    min_train_days: int = 30,
) -> dict[str, Any]:
    """
    Walk-forward validation with retraining after each fold.
    Simulates production weekly retraining.

    Split strategy:
      Fold 1: train on [day 0..30], test on [day 30..60]
      Fold 2: train on [day 0..60], test on [day 60..90]
      ...

    This is the most realistic offline evaluation for time-series models.
    """
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    ts = pd.to_datetime(df[timestamp_col], utc=True)
    total_days = (ts.max() - ts.min()).days

    step_days = (total_days - min_train_days) // n_folds
    results: list[dict] = []

    for fold in range(n_folds):
        train_end_day = min_train_days + fold * step_days
        test_end_day = train_end_day + step_days

        start_ts = ts.min()
        train_mask = ts < (start_ts + pd.Timedelta(days=train_end_day))
        test_mask = (ts >= (start_ts + pd.Timedelta(days=train_end_day))) & \
                    (ts < (start_ts + pd.Timedelta(days=test_end_day)))

        X_train = df.loc[train_mask, feature_cols].fillna(0).values
        y_train = df.loc[train_mask, label_col].values
        X_test = df.loc[test_mask, feature_cols].fillna(0).values
        y_test = df.loc[test_mask, label_col].values

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        model = model_factory(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
            y_pred = (y_score >= 0.5).astype(int)
        else:
            y_score = None
            y_pred = model.predict(X_test)

        fold_metrics = evaluate(y_test, y_pred, y_score=y_score)
        fold_metrics["fold"] = fold + 1
        fold_metrics["train_days"] = train_end_day
        fold_metrics["test_days"] = step_days
        results.append(fold_metrics)

        log.info(
            "walk_forward_fold",
            fold=fold + 1,
            train_days=train_end_day,
            train_samples=len(X_train),
            test_samples=len(X_test),
            pr_auc=fold_metrics.get("pr_auc"),
        )

    return {"walk_forward_folds": results}


# ── Bootstrap confidence intervals ────────────────────────────────────────────

def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> dict[str, float]:
    """
    Compute bootstrap confidence interval for a scalar metric.

    Parameters
    ----------
    y_true     : ground truth
    y_score    : predicted scores
    metric_fn  : callable(y_true, y_score) → float
    n_bootstrap: number of bootstrap samples
    ci         : confidence level (0.95 = 95% CI)

    Returns
    -------
    {"mean": ..., "lower": ..., "upper": ..., "std": ...}

    Interview talking point:
      "Bootstrap CI is model-free: we resample with replacement, compute
       the metric each time, and take the percentiles. It works for ANY
       metric — PR-AUC, F-beta, cost-weighted score — without assuming
       any distributional form. This is statistically rigorous and easy
       to explain to non-technical stakeholders."
    """
    rng = np.random.default_rng(random_seed)
    scores: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_t = y_true[idx]
        y_s = y_score[idx]
        if len(np.unique(y_t)) < 2:
            continue
        scores.append(metric_fn(y_t, y_s))

    scores_arr = np.array(scores)
    alpha = 1 - ci
    return {
        "mean": round(float(np.mean(scores_arr)), 4),
        "std": round(float(np.std(scores_arr)), 4),
        "lower": round(float(np.percentile(scores_arr, 100 * alpha / 2)), 4),
        "upper": round(float(np.percentile(scores_arr, 100 * (1 - alpha / 2))), 4),
        "n_bootstrap": n_bootstrap,
        "ci": ci,
    }

