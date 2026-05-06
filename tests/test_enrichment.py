"""
Tests for enrichment module.
"""
import pandas as pd
import pytest

from src.enrichment import normalize_merchant, enrich_transactions


def make_df(n: int = 10) -> pd.DataFrame:
    import datetime
    rows = []
    for i in range(n):
        rows.append({
            "transaction_id": f"tx_{i}",
            "user_id": "user_1" if i < 7 else "user_2",
            "timestamp": (datetime.datetime(2024, 1, 1) + datetime.timedelta(hours=i * 3)).isoformat(),
            "amount": float(100 + i * 10),
            "merchant": "Amazon" if i % 2 == 0 else "Walmart",
        })
    return pd.DataFrame(rows)


class TestNormalizeMerchant:
    def test_amazon_variants(self):
        assert normalize_merchant("Amazon.com") == "amazon"
        assert normalize_merchant("AMAZON PRIME") == "amazon"
        assert normalize_merchant("amzn") == "amazon"

    def test_walmart_variants(self):
        assert normalize_merchant("WALMART #1234") == "walmart"
        assert normalize_merchant("Walmart Supercenter") == "walmart"

    def test_unknown_non_string(self):
        assert normalize_merchant(None) == "unknown"
        assert normalize_merchant(123) == "unknown"

    def test_unknown_merchant(self):
        result = normalize_merchant("Random Coffee Shop")
        assert isinstance(result, str)
        assert result == "random coffee shop"

    def test_strips_whitespace(self):
        result = normalize_merchant("  starbucks  ")
        assert result == "starbucks"


class TestEnrichTransactions:
    def test_returns_copy(self):
        df = make_df()
        enriched = enrich_transactions(df)
        assert enriched is not df

    def test_adds_expected_columns(self):
        df = make_df()
        enriched = enrich_transactions(df)
        for col in ["merchant_clean", "hour", "day_of_week", "is_night", "is_weekend",
                    "amount_mean_7", "amount_std_7"]:
            assert col in enriched.columns, f"Missing column: {col}"

    def test_is_night_range(self):
        df = make_df()
        enriched = enrich_transactions(df)
        assert enriched["is_night"].isin([0, 1]).all()

    def test_rolling_mean_non_negative(self):
        df = make_df()
        enriched = enrich_transactions(df)
        assert (enriched["amount_mean_7"] >= 0).all()

    def test_rolling_std_non_negative(self):
        df = make_df()
        enriched = enrich_transactions(df)
        assert (enriched["amount_std_7"] >= 0).all()
