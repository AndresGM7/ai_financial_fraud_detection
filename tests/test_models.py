"""
Tests for statistical, unsupervised, and supervised detection models.
"""
import datetime

import numpy as np
import pandas as pd
import pytest

from src.enrichment import enrich_transactions
from src.features import build_features
from src.models.statistical import zscore_rule, night_rule, ewma_anomaly, combine_statistical
from src.models.unsupervised import train_isolation_forest, predict_iforest, fit_predict_lof, build_scaler
from src.models.supervised import train_logreg, predict_logreg, predict_proba_logreg


def make_featured_df(n: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "transaction_id": f"tx_{i}",
            "user_id": "user_1" if i < 20 else "user_2",
            "timestamp": (datetime.datetime(2024, 1, 1) + datetime.timedelta(hours=i)).isoformat(),
            "amount": float(100 + i * 5),
            "merchant": "Amazon",
        })
    df = pd.DataFrame(rows)
    enriched = enrich_transactions(df)
    return build_features(enriched)


class TestStatisticalModels:
    def setup_method(self):
        self.df = make_featured_df()

    def test_zscore_rule_binary(self):
        result = zscore_rule(self.df)
        assert result.isin([0, 1]).all()
        assert len(result) == len(self.df)

    def test_night_rule_binary(self):
        result = night_rule(self.df)
        assert result.isin([0, 1]).all()

    def test_ewma_anomaly_binary(self):
        result = ewma_anomaly(self.df)
        assert result.isin([0, 1]).all()

    def test_combine_statistical_binary(self):
        result = combine_statistical(self.df)
        assert result.isin([0, 1]).all()
        assert len(result) == len(self.df)

    def test_high_amount_triggers_zscore(self):
        df = self.df.copy()
        df.loc[df.index[-1], "z_amount"] = 10.0
        result = zscore_rule(df)
        assert result.iloc[-1] == 1


class TestUnsupervisedModels:
    def setup_method(self):
        self.df = make_featured_df(n=50)
        from src.features import get_feature_columns
        cols = get_feature_columns()
        self.X = self.df[cols].fillna(0).values

    def test_iforest_train_and_predict(self):
        model = train_isolation_forest(self.X, n_estimators=50)
        preds = predict_iforest(model, self.X)
        assert set(preds).issubset({0, 1})
        assert len(preds) == len(self.X)

    def test_lof_fit_predict(self):
        preds = fit_predict_lof(self.X, n_neighbors=5)
        assert set(preds).issubset({0, 1})
        assert len(preds) == len(self.X)

    def test_build_scaler(self):
        scaler = build_scaler(self.X)
        X_scaled = scaler.transform(self.X)
        assert X_scaled.shape == self.X.shape


class TestSupervisedModels:
    def setup_method(self):
        self.df = make_featured_df(n=60)
        from src.features import get_feature_columns
        cols = get_feature_columns()
        self.X = self.df[cols].fillna(0).values
        # Synthetic labels: last 10 are "fraud"
        self.y = np.zeros(len(self.X), dtype=int)
        self.y[-10:] = 1

    def test_logreg_train_and_predict(self):
        model = train_logreg(self.X, self.y)
        preds = predict_logreg(model, self.X)
        assert set(preds).issubset({0, 1})
        assert len(preds) == len(self.X)

    def test_logreg_predict_proba_range(self):
        model = train_logreg(self.X, self.y)
        proba = predict_proba_logreg(model, self.X)
        assert ((proba >= 0) & (proba <= 1)).all()
