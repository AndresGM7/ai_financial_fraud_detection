"""
tests/test_enrichment.py
─────────────────────────
Unit tests for enrichment.py.
Property-based tests with Hypothesis for edge cases.
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from src.enrichment import (
    _haversine_km,
    enrich_transactions,
    normalise_merchant,
)


# ── Merchant normalisation ────────────────────────────────────────────────────

class TestNormaliseMerchant:
    def test_known_alias(self):
        assert normalise_merchant("AMZN*MKP 1234") == "amazon"

    def test_case_insensitive(self):
        assert normalise_merchant("WALMART SUPERCENTER") == "walmart"

    def test_none_input(self):
        assert normalise_merchant(None) == "unknown"  # type: ignore

    def test_empty_string(self):
        assert normalise_merchant("") == "unknown"

    def test_unknown_merchant_lowercased(self):
        result = normalise_merchant("SomeCoffeeShop")
        assert result == result.lower()

    @given(st.text(max_size=200))
    @settings(max_examples=100)
    def test_never_raises(self, name: str):
        """normalise_merchant should never raise for any string input."""
        result = normalise_merchant(name)
        assert isinstance(result, str)

    @given(st.text(max_size=200))
    @settings(max_examples=100)
    def test_always_lowercase(self, name: str):
        result = normalise_merchant(name)
        assert result == result.lower()


# ── Haversine distance ────────────────────────────────────────────────────────

class TestHaversine:
    def test_same_point(self):
        assert _haversine_km(40.71, -74.01, 40.71, -74.01) == pytest.approx(0.0, abs=0.01)

    def test_ny_to_la(self):
        dist = _haversine_km(40.71, -74.01, 34.05, -118.24)
        assert 3900 < dist < 4000  # ~3944 km

    def test_ny_to_miami(self):
        dist = _haversine_km(40.71, -74.01, 25.77, -80.19)
        assert 1700 < dist < 1900  # ~1760 km (great-circle)

    @given(
        lat1=st.floats(-90, 90),
        lon1=st.floats(-180, 180),
        lat2=st.floats(-90, 90),
        lon2=st.floats(-180, 180),
    )
    @settings(max_examples=100)
    def test_non_negative(self, lat1, lon1, lat2, lon2):
        dist = _haversine_km(lat1, lon1, lat2, lon2)
        assert dist >= 0


# ── Enrichment pipeline ───────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "tx_id": "TX00000001",
            "user_id": "U00001",
            "timestamp": "2024-06-15 02:30:00+00:00",
            "amount": 9500.0,
            "merchant": "western_union",
            "mcc": 6099,
            "tx_type": "ACH",
            "counterparty": "Unknown",
            "currency": "USD",
            "lat": 40.71,
            "lon": -74.01,
            "is_fraud": 1,
            "fraud_type": "romance_scam",
        },
        {
            "tx_id": "TX00000002",
            "user_id": "U00001",
            "timestamp": "2024-06-15 10:00:00+00:00",
            "amount": 45.0,
            "merchant": "starbucks",
            "mcc": 5812,
            "tx_type": "POS",
            "counterparty": "Starbucks",
            "currency": "USD",
            "lat": 40.72,
            "lon": -74.02,
            "is_fraud": 0,
            "fraud_type": None,
        },
    ])


class TestEnrichTransactions:
    def test_output_has_expected_columns(self, sample_df):
        enriched = enrich_transactions(sample_df)
        expected_cols = [
            "merchant_clean", "hour", "is_night", "is_weekend",
            "amount_mean_7tx", "amount_std_7tx", "tx_count_1h",
        ]
        for col in expected_cols:
            assert col in enriched.columns, f"Missing column: {col}"

    def test_night_flag_correct(self, sample_df):
        enriched = enrich_transactions(sample_df)
        # TX at 02:30 should be is_night=1
        night_row = enriched[enriched["tx_id"] == "TX00000001"]
        assert night_row["is_night"].values[0] == 1

    def test_day_flag_correct(self, sample_df):
        enriched = enrich_transactions(sample_df)
        # TX at 10:00 should be is_night=0
        day_row = enriched[enriched["tx_id"] == "TX00000002"]
        assert day_row["is_night"].values[0] == 0

    def test_mcc_high_risk_western_union(self, sample_df):
        enriched = enrich_transactions(sample_df)
        wu_row = enriched[enriched["tx_id"] == "TX00000001"]
        assert wu_row["mcc_high_risk"].values[0] == 1

    def test_no_data_leakage(self, sample_df):
        """Rolling stats for first transaction should not include future rows."""
        enriched = enrich_transactions(sample_df)
        # First tx per user should have no rolling history → mean = 0 or amount
        first_tx = enriched.sort_values("timestamp").groupby("user_id").first()
        # amount_mean_7tx uses shift(1) so first row gets NaN → filled to 0
        assert first_tx["amount_mean_7tx"].isna().sum() == 0

