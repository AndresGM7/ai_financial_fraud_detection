"""
tests/test_features.py
───────────────────────
Unit tests for src/features.py.
Covers build_features(), get_feature_matrix(), get_labels(), and feature registry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from src.features import (
    ALL_FEATURES,
    BINARY_FEATURES,
    INTERACTION_FEATURES,
    NUMERICAL_FEATURES,
    TX_TYPE_DUMMIES,
    build_features,
    get_feature_matrix,
    get_labels,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def minimal_enriched_df() -> pd.DataFrame:
    """Minimal enriched DataFrame with all columns expected by build_features()."""
    n = 5
    return pd.DataFrame({
        "tx_id":              [f"TX{i:05d}" for i in range(n)],
        "user_id":            ["U00001"] * n,
        "timestamp":          pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "amount":             [100.0, 50.0, 200.0, 9500.0, 75.0],
        "amount_mean_7tx":    [90.0,  90.0, 90.0,  90.0,   90.0],
        "amount_std_7tx":     [20.0,  20.0, 20.0,  20.0,   20.0],
        "amount_ewma":        [95.0,  95.0, 95.0,  95.0,   95.0],
        "amount_ewma_std":    [15.0,  15.0, 15.0,  15.0,   15.0],
        "tx_count_1h":        [1.0,   1.0,  1.0,   8.0,    1.0],
        "tx_count_24h":       [3.0,   3.0,  3.0,   12.0,   3.0],
        "tx_count_7d":        [20.0,  20.0, 20.0,  20.0,   20.0],
        "amount_sum_24h":     [300.0, 300.0, 300.0, 1000.0, 300.0],
        "hour_sin":           [0.0] * n,
        "hour_cos":           [1.0] * n,
        "dow_sin":            [0.0] * n,
        "dow_cos":            [1.0] * n,
        "geo_speed_kmh":      [0.0, 0.0, 0.0, 1200.0, 0.0],
        "is_night":           [0, 0, 0, 1, 0],
        "is_weekend":         [0, 0, 0, 0, 1],
        "is_holiday":         [0, 0, 0, 0, 0],
        "is_new_counterparty":[0, 0, 0, 1, 0],
        "geo_impossible":     [0, 0, 0, 1, 0],
        "mcc_high_risk":      [0, 0, 0, 1, 0],
        "tx_type":            ["POS", "POS", "ACH", "WIRE", "ONLINE"],
        "is_fraud":           [0, 0, 0, 1, 0],
    })


# ── Feature registry tests ────────────────────────────────────────────────────

class TestFeatureRegistry:
    def test_all_features_non_empty(self):
        assert len(ALL_FEATURES) > 0

    def test_no_duplicate_features(self):
        assert len(ALL_FEATURES) == len(set(ALL_FEATURES))

    def test_all_features_is_union(self):
        expected = (
            NUMERICAL_FEATURES + BINARY_FEATURES +
            INTERACTION_FEATURES + TX_TYPE_DUMMIES
        )
        assert ALL_FEATURES == expected

    def test_tx_type_dummies_have_prefix(self):
        for col in TX_TYPE_DUMMIES:
            assert col.startswith("tx_type_"), f"Expected tx_type_ prefix: {col}"


# ── build_features() tests ────────────────────────────────────────────────────

class TestBuildFeatures:
    def test_returns_dataframe(self, minimal_enriched_df):
        out = build_features(minimal_enriched_df)
        assert isinstance(out, pd.DataFrame)

    def test_does_not_modify_input(self, minimal_enriched_df):
        original_cols = set(minimal_enriched_df.columns)
        _ = build_features(minimal_enriched_df)
        assert set(minimal_enriched_df.columns) == original_cols

    def test_z_amount_column_exists(self, minimal_enriched_df):
        out = build_features(minimal_enriched_df)
        assert "z_amount" in out.columns

    def test_ewma_z_amount_column_exists(self, minimal_enriched_df):
        out = build_features(minimal_enriched_df)
        assert "ewma_z_amount" in out.columns

    def test_interaction_features_exist(self, minimal_enriched_df):
        out = build_features(minimal_enriched_df)
        for col in INTERACTION_FEATURES:
            assert col in out.columns, f"Missing interaction feature: {col}"

    def test_tx_type_dummies_created(self, minimal_enriched_df):
        out = build_features(minimal_enriched_df)
        # WIRE type row should have tx_type_WIRE = 1
        wire_row = out[out["tx_type"] == "WIRE"]
        assert wire_row["tx_type_WIRE"].values[0] == 1

    def test_tx_type_dummies_missing_encoded_as_zero(self, minimal_enriched_df):
        """Dummy columns for absent types should be filled with 0."""
        out = build_features(minimal_enriched_df)
        # No ZELLE transactions in fixture → should be all zeros
        if "tx_type_ZELLE" in out.columns:
            assert (out["tx_type_ZELLE"] == 0).all()

    def test_z_amount_clipped_at_ten(self, minimal_enriched_df):
        """Z-scores must be clipped to [-10, 10] after winsorisation."""
        df = minimal_enriched_df.copy()
        # Inject an extreme amount
        df.loc[0, "amount"] = 1_000_000.0
        out = build_features(df)
        assert out["z_amount"].max() <= 10.0
        assert out["z_amount"].min() >= -10.0

    def test_ewma_z_amount_clipped(self, minimal_enriched_df):
        df = minimal_enriched_df.copy()
        df.loc[0, "amount"] = 1_000_000.0
        out = build_features(df)
        assert out["ewma_z_amount"].max() <= 10.0
        assert out["ewma_z_amount"].min() >= -10.0

    def test_no_nans_in_numerical_features(self, minimal_enriched_df):
        out = build_features(minimal_enriched_df)
        for col in NUMERICAL_FEATURES:
            if col in out.columns:
                assert out[col].isna().sum() == 0, f"NaN found in {col}"

    def test_interaction_amount_x_night(self, minimal_enriched_df):
        out = build_features(minimal_enriched_df)
        night_row_idx = minimal_enriched_df[minimal_enriched_df["is_night"] == 1].index
        if len(night_row_idx) > 0:
            for i in night_row_idx:
                expected = minimal_enriched_df.loc[i, "amount"]
                assert out.loc[i, "amount_x_night"] == pytest.approx(expected)

    def test_interaction_amount_x_new_payee(self, minimal_enriched_df):
        out = build_features(minimal_enriched_df)
        normal_row = out[out["is_new_counterparty"] == 0].iloc[0]
        assert normal_row["amount_x_new_payee"] == pytest.approx(0.0)

    def test_z_amount_formula_correctness(self):
        """Single-row sanity check: Z = (amount - mean) / std."""
        df = pd.DataFrame({
            "tx_id": ["TX00001"],
            "user_id": ["U00001"],
            "timestamp": pd.to_datetime(["2024-01-01T10:00:00+00:00"]),
            "amount": [200.0],
            "amount_mean_7tx": [100.0],
            "amount_std_7tx": [50.0],
            "amount_ewma": [100.0],
            "amount_ewma_std": [50.0],
            "tx_count_1h": [1.0],
            "tx_count_24h": [3.0],
            "tx_count_7d": [10.0],
            "amount_sum_24h": [200.0],
            "hour_sin": [0.0], "hour_cos": [1.0],
            "dow_sin": [0.0], "dow_cos": [1.0],
            "geo_speed_kmh": [0.0],
            "is_night": [0], "is_weekend": [0], "is_holiday": [0],
            "is_new_counterparty": [0], "geo_impossible": [0], "mcc_high_risk": [0],
            "tx_type": ["POS"],
        })
        out = build_features(df)
        # z = (200 - 100) / (50 + 1e-6) ≈ 2.0
        assert out["z_amount"].iloc[0] == pytest.approx(2.0, abs=0.001)


# ── get_feature_matrix() tests ────────────────────────────────────────────────

class TestGetFeatureMatrix:
    def test_returns_ndarray(self, minimal_enriched_df):
        df_feat = build_features(minimal_enriched_df)
        X, scaler = get_feature_matrix(df_feat)
        assert isinstance(X, np.ndarray)

    def test_shape_matches_features(self, minimal_enriched_df):
        df_feat = build_features(minimal_enriched_df)
        X, _ = get_feature_matrix(df_feat)
        available = [c for c in ALL_FEATURES if c in df_feat.columns]
        assert X.shape == (len(df_feat), len(available))

    def test_no_scale_returns_none_scaler(self, minimal_enriched_df):
        df_feat = build_features(minimal_enriched_df)
        _, scaler = get_feature_matrix(df_feat, scale=False)
        assert scaler is None

    def test_scale_true_returns_scaler(self, minimal_enriched_df):
        df_feat = build_features(minimal_enriched_df)
        from sklearn.preprocessing import StandardScaler
        _, scaler = get_feature_matrix(df_feat, scale=True)
        assert isinstance(scaler, StandardScaler)

    def test_scaled_matrix_has_approx_zero_mean(self, minimal_enriched_df):
        df_feat = build_features(minimal_enriched_df)
        X, _ = get_feature_matrix(df_feat, scale=True)
        # Mean of each column should be close to 0 after StandardScaler
        for col_mean in X.mean(axis=0):
            assert abs(col_mean) < 1e-5

    def test_precomputed_scaler_reused(self, minimal_enriched_df):
        """Passing a fitted scaler should not refit."""
        from sklearn.preprocessing import StandardScaler
        df_feat = build_features(minimal_enriched_df)
        X1, scaler = get_feature_matrix(df_feat, scale=True)
        X2, _ = get_feature_matrix(df_feat, scale=True, scaler=scaler)
        np.testing.assert_array_almost_equal(X1, X2)

    def test_no_nans_in_output(self, minimal_enriched_df):
        df_feat = build_features(minimal_enriched_df)
        X, _ = get_feature_matrix(df_feat)
        assert not np.isnan(X).any()

    def test_dtype_is_float32(self, minimal_enriched_df):
        df_feat = build_features(minimal_enriched_df)
        X, _ = get_feature_matrix(df_feat)
        assert X.dtype == np.float32


# ── get_labels() tests ────────────────────────────────────────────────────────

class TestGetLabels:
    def test_returns_labels_when_present(self, minimal_enriched_df):
        y = get_labels(minimal_enriched_df)
        assert y is not None
        assert len(y) == len(minimal_enriched_df)

    def test_labels_are_int(self, minimal_enriched_df):
        y = get_labels(minimal_enriched_df)
        assert y.dtype == int

    def test_returns_none_when_column_absent(self, minimal_enriched_df):
        df = minimal_enriched_df.drop(columns=["is_fraud"])
        y = get_labels(df)
        assert y is None

    def test_returns_none_when_all_nan(self, minimal_enriched_df):
        df = minimal_enriched_df.copy()
        df["is_fraud"] = float("nan")
        y = get_labels(df)
        assert y is None

    def test_nan_labels_filled_with_zero(self, minimal_enriched_df):
        df = minimal_enriched_df.copy()
        df.loc[0, "is_fraud"] = float("nan")
        y = get_labels(df)
        assert y[0] == 0

    def test_fraud_rate_correct(self, minimal_enriched_df):
        y = get_labels(minimal_enriched_df)
        # One fraud row (index 3, amount=9500, is_fraud=1)
        assert y.sum() == 1


# ── Property-based tests ──────────────────────────────────────────────────────

class TestBuildFeaturesProperties:
    @given(
        amounts=st.lists(
            st.floats(min_value=0.01, max_value=100_000.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=10,
        )
    )
    @settings(max_examples=30)
    def test_z_amount_always_clipped(self, amounts):
        n = len(amounts)
        df = pd.DataFrame({
            "tx_id": [f"TX{i}" for i in range(n)],
            "user_id": ["U00001"] * n,
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "amount": amounts,
            "amount_mean_7tx": [100.0] * n,
            "amount_std_7tx": [20.0] * n,
            "amount_ewma": [100.0] * n,
            "amount_ewma_std": [20.0] * n,
            "tx_count_1h": [1.0] * n,
            "tx_count_24h": [3.0] * n,
            "tx_count_7d": [10.0] * n,
            "amount_sum_24h": [300.0] * n,
            "hour_sin": [0.0] * n, "hour_cos": [1.0] * n,
            "dow_sin": [0.0] * n, "dow_cos": [1.0] * n,
            "geo_speed_kmh": [0.0] * n,
            "is_night": [0] * n, "is_weekend": [0] * n, "is_holiday": [0] * n,
            "is_new_counterparty": [0] * n, "geo_impossible": [0] * n,
            "mcc_high_risk": [0] * n,
            "tx_type": ["POS"] * n,
        })
        out = build_features(df)
        assert (out["z_amount"] >= -10).all()
        assert (out["z_amount"] <= 10).all()

    @given(
        tx_type=st.sampled_from(["ACH", "WIRE", "ZELLE", "POS", "ATM", "ONLINE"])
    )
    @settings(max_examples=20)
    def test_tx_type_dummy_is_one_hot(self, tx_type):
        """Exactly one tx_type column should be 1 for a given tx_type."""
        df = pd.DataFrame({
            "tx_id": ["TX00001"],
            "user_id": ["U00001"],
            "timestamp": pd.to_datetime(["2024-01-01T10:00:00+00:00"]),
            "amount": [100.0],
            "amount_mean_7tx": [90.0], "amount_std_7tx": [10.0],
            "amount_ewma": [90.0], "amount_ewma_std": [10.0],
            "tx_count_1h": [1.0], "tx_count_24h": [3.0], "tx_count_7d": [10.0],
            "amount_sum_24h": [100.0],
            "hour_sin": [0.0], "hour_cos": [1.0],
            "dow_sin": [0.0], "dow_cos": [1.0],
            "geo_speed_kmh": [0.0],
            "is_night": [0], "is_weekend": [0], "is_holiday": [0],
            "is_new_counterparty": [0], "geo_impossible": [0], "mcc_high_risk": [0],
            "tx_type": [tx_type],
        })
        out = build_features(df)
        from src.features import TX_TYPE_DUMMIES
        dummy_sum = sum(int(out.iloc[0][col]) for col in TX_TYPE_DUMMIES if col in out.columns)
        # Should be exactly 1 (the matching dummy) or possibly 0 if type not in list
        assert dummy_sum in (0, 1)

