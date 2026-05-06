"""
Tests for evaluation modules.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import evaluate
from src.evaluation.cross_validation import time_based_splits, cross_validate_model
from src.evaluation.error_analysis import error_analysis


class TestEvaluateMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 0, 1, 1])
        result = evaluate(y, y)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        result = evaluate(y_true, y_pred)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_with_probabilities(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        result = evaluate(y_true, y_pred, y_score)
        assert "roc_auc" in result
        assert result["roc_auc"] is not None
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        result = evaluate(y_true, y_pred)
        cm = result["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_support_counts(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        result = evaluate(y_true, y_pred)
        assert result["support_positive"] == 3
        assert result["support_total"] == 5


class TestTimeSplits:
    def _make_df(self, n: int = 50) -> pd.DataFrame:
        import datetime
        rows = [
            {
                "timestamp": (
                    datetime.datetime(2024, 1, 1) + datetime.timedelta(hours=i)
                ).isoformat()
            }
            for i in range(n)
        ]
        return pd.DataFrame(rows)

    def test_returns_correct_number_of_splits(self):
        df = self._make_df(50)
        splits = time_based_splits(df, n_splits=5)
        assert len(splits) == 5

    def test_train_before_test(self):
        df = self._make_df(60)
        splits = time_based_splits(df, n_splits=4)
        for train_idx, test_idx in splits:
            assert max(train_idx) < min(test_idx)


class TestErrorAnalysis:
    def test_fp_fn_counts(self):
        df = pd.DataFrame({"amount": [100, 200, 300, 400], "is_night": [0, 1, 0, 1]})
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1])
        result = error_analysis(df, y_true, y_pred)
        assert result["summary"]["false_positive_count"] == 1
        assert result["summary"]["false_negative_count"] == 1

    def test_perfect_no_errors(self):
        df = pd.DataFrame({"amount": [100, 200]})
        y = np.array([0, 1])
        result = error_analysis(df, y, y)
        assert result["summary"]["false_positive_count"] == 0
        assert result["summary"]["false_negative_count"] == 0
