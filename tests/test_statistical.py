"""
tests/test_statistical.py
──────────────────────────
Unit tests for models/statistical.py detectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.statistical import (
    NightLargeWireRule,
    RapidSuccessionRule,
    RulesEngine,
    StructuringRule,
    cusum_scores,
    statistical_score,
    zscore_flag,
    zscore_score,
)


@pytest.fixture
def base_df() -> pd.DataFrame:
    n = 20
    return pd.DataFrame({
        "tx_id": [f"TX{i:05d}" for i in range(n)],
        "user_id": ["U00001"] * n,
        "amount": [100.0] * n,
        "amount_mean_7tx": [100.0] * n,
        "amount_std_7tx": [10.0] * n,
        "amount_ewma": [100.0] * n,
        "amount_ewma_std": [10.0] * n,
        "z_amount": [0.0] * n,
        "ewma_z_amount": [0.0] * n,
        "is_night": [0] * n,
        "tx_count_1h": [1.0] * n,
        "tx_count_24h": [3.0] * n,
        "tx_count_7d": [20.0] * n,
        "amount_sum_24h": [300.0] * n,
        "tx_type": ["POS"] * n,
        "tx_type_WIRE": [0] * n,
        "tx_type_ACH": [0] * n,
        "tx_type_ATM": [0] * n,
        "tx_type_ZELLE": [0] * n,
        "tx_type_ONLINE": [0] * n,
        "tx_type_POS": [1] * n,
        "is_new_counterparty": [0] * n,
        "geo_impossible": [0] * n,
        "mcc_high_risk": [0] * n,
    })


class TestZscoreFlag:
    def test_normal_transaction_not_flagged(self, base_df):
        flags = zscore_flag(base_df, threshold=3.0)
        assert flags.sum() == 0

    def test_extreme_amount_flagged(self, base_df):
        df = base_df.copy()
        df.loc[0, "amount"] = 5000.0  # extreme outlier
        df.loc[0, "z_amount"] = 49.0  # (5000-100)/10 = 490, clipped to 10 in features but raw here
        # Use non-log mode for deterministic test
        flags = zscore_flag(df, threshold=3.0, use_log=False)
        assert flags.iloc[0] == 1

    def test_score_range(self, base_df):
        scores = zscore_score(base_df)
        assert scores.between(0, 1).all()


class TestCUSUM:
    def test_stable_series_low_scores(self):
        stable = pd.Series(np.ones(50) * 100.0)
        scores = cusum_scores(stable)
        assert scores.max() <= 0.1  # stable → near-zero CUSUM

    def test_drifting_series_high_scores(self):
        # Gradual upward drift from 100 to 1000
        drifting = pd.Series(np.linspace(100, 1000, 50))
        scores = cusum_scores(drifting)
        assert scores.iloc[-1] > 0.5  # should accumulate high score


class TestRulesEngine:
    def test_no_rules_triggered_normal(self, base_df):
        engine = RulesEngine()
        result = engine.apply(base_df)
        assert result["rules_score"].max() < 0.3

    def test_night_large_wire_rule(self, base_df):
        df = base_df.copy()
        df.loc[0, "is_night"] = 1
        df.loc[0, "tx_type_WIRE"] = 1
        df.loc[0, "amount"] = 500.0  # > 3× mean of 100
        df.loc[0, "is_new_counterparty"] = 1
        rule = NightLargeWireRule()
        flags = rule.apply(df)
        assert flags.iloc[0] == 1
        assert flags.iloc[1] == 0

    def test_rapid_succession_rule(self, base_df):
        df = base_df.copy()
        df.loc[0, "tx_count_1h"] = 10.0
        rule = RapidSuccessionRule()
        flags = rule.apply(df)
        assert flags.iloc[0] == 1

    def test_structuring_rule(self, base_df):
        df = base_df.copy()
        df.loc[0, "tx_type_ATM"] = 1
        df.loc[0, "amount"] = 8500.0
        rule = StructuringRule()
        flags = rule.apply(df)
        assert flags.iloc[0] == 1

    def test_rules_score_bounded(self, base_df):
        engine = RulesEngine()
        result = engine.apply(base_df)
        assert result["rules_score"].between(0, 1).all()


class TestStatisticalScore:
    def test_score_bounded(self, base_df):
        scores = statistical_score(base_df)
        assert scores.between(0, 1).all()

    def test_anomalous_tx_higher_score(self, base_df):
        df_normal = base_df.copy()
        df_anomalous = base_df.copy()
        df_anomalous.loc[0, "z_amount"] = 8.0
        df_anomalous.loc[0, "ewma_z_amount"] = 7.0
        df_anomalous.loc[0, "tx_count_1h"] = 10.0
        df_anomalous.loc[0, "is_night"] = 1

        normal_score = statistical_score(df_normal).iloc[0]
        anomalous_score = statistical_score(df_anomalous).iloc[0]
        assert anomalous_score > normal_score

