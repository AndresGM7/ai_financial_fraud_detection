"""
Tests for feature engineering module.
"""
import datetime

import pandas as pd
import pytest

from src.enrichment import enrich_transactions
from src.features import build_features, get_feature_columns


def make_enriched_df(n: int = 15) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "transaction_id": f"tx_{i}",
            "user_id": "user_1" if i < 10 else "user_2",
            "timestamp": (datetime.datetime(2024, 1, 1) + datetime.timedelta(hours=i * 2)).isoformat(),
            "amount": float(50 + i * 20),
            "merchant": ["Amazon", "Walmart", "Target"][i % 3],
        })
    df = pd.DataFrame(rows)
    return enrich_transactions(df)


class TestBuildFeatures:
    def test_returns_copy(self):
        df = make_enriched_df()
        featured = build_features(df)
        assert featured is not df

    def test_z_amount_column_exists(self):
        df = make_enriched_df()
        featured = build_features(df)
        assert "z_amount" in featured.columns

    def test_is_new_merchant_binary(self):
        df = make_enriched_df()
        featured = build_features(df)
        assert featured["is_new_merchant"].isin([0, 1]).all()

    def test_amount_ratio_non_negative(self):
        df = make_enriched_df()
        featured = build_features(df)
        assert (featured["amount_ratio"] >= 0).all()

    def test_tx_count_24h_positive(self):
        df = make_enriched_df()
        featured = build_features(df)
        assert (featured["tx_count_24h"] >= 1).all()

    def test_feature_columns_present(self):
        df = make_enriched_df()
        featured = build_features(df)
        for col in get_feature_columns():
            assert col in featured.columns, f"Missing feature column: {col}"


class TestGetFeatureColumns:
    def test_returns_list(self):
        cols = get_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_contains_key_features(self):
        cols = get_feature_columns()
        for key in ["amount", "z_amount", "is_new_merchant"]:
            assert key in cols
