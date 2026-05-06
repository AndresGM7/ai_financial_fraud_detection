"""
Cross-validation: time-aware cross-validation for fraud detection models.

Note: Standard k-fold CV risks temporal leakage in time-series data.
We implement time-based splits (walk-forward validation) as the preferred approach.
"""
import logging
from typing import Callable, List

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate

logger = logging.getLogger(__name__)


def time_based_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    timestamp_col: str = "timestamp",
) -> List[tuple]:
    """
    Generate (train_idx, test_idx) pairs using time-based walk-forward splits.

    This avoids temporal leakage: training data always precedes test data.
    Each fold trains on the first i/n_splits of data and tests on the next chunk.

    Args:
        df: DataFrame sorted by timestamp.
        n_splits: Number of walk-forward splits.
        timestamp_col: Name of the timestamp column.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df)
    fold_size = n // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        test_end = min((i + 1) * fold_size, n)
        if test_end > train_end:
            splits.append((
                np.arange(0, train_end),
                np.arange(train_end, test_end),
            ))
    return splits


def cross_validate_model(
    model_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    n_splits: int = 5,
) -> List[dict]:
    """
    Run time-based cross-validation for a model.

    Args:
        model_fn: Function that takes (X_train, y_train) and returns a fitted model
                  with a .predict() method.
        X: Feature matrix.
        y: Labels.
        df: Original DataFrame (used for time-based splitting).
        n_splits: Number of walk-forward folds.

    Returns:
        List of metric dicts, one per fold.
    """
    splits = time_based_splits(df, n_splits=n_splits)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2:
            logger.warning("Fold %d: only one class in training set, skipping", fold_idx)
            continue

        model = model_fn(X_train, y_train)
        y_pred = model.predict(X_test)

        y_score = None
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_pred, y_score)
        metrics["fold"] = fold_idx
        fold_results.append(metrics)
        logger.info(
            "Fold %d — precision=%.3f recall=%.3f f1=%.3f",
            fold_idx,
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )

    return fold_results
