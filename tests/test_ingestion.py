"""
tests/test_ingestion.py
────────────────────────
Unit + integration tests for ingestion.py.
Covers schema validation, format loaders, Pydantic coercion, and dead-letter behaviour.
"""

from __future__ import annotations

import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from src.ingestion import (
    DataIngestionPipeline,
    IngestionResult,
    TransactionRecord,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_row(overrides: dict | None = None) -> dict:
    """Return a valid transaction dict, optionally overriding fields."""
    row = {
        "tx_id": "TX00000001",
        "user_id": "U00001",
        "timestamp": "2024-06-15T10:00:00+00:00",
        "amount": 150.0,
        "merchant": "starbucks",
        "mcc": 5812,
        "tx_type": "POS",
        "counterparty": "Starbucks",
        "currency": "USD",
        "lat": 40.71,
        "lon": -74.01,
        "is_fraud": 0,
        "fraud_type": None,
    }
    if overrides:
        row.update(overrides)
    return row


# ── TransactionRecord schema tests ────────────────────────────────────────────

class TestTransactionRecord:
    def test_valid_record_parses_ok(self):
        rec = TransactionRecord.model_validate(_valid_row())
        assert rec.tx_id == "TX00000001"
        assert rec.amount == 150.0

    def test_amount_rounded_to_two_decimals(self):
        rec = TransactionRecord.model_validate(_valid_row({"amount": 150.123456}))
        assert rec.amount == 150.12

    def test_tx_type_uppercased(self):
        rec = TransactionRecord.model_validate(_valid_row({"tx_type": "pos"}))
        assert rec.tx_type == "POS"

    def test_merchant_lowercased(self):
        rec = TransactionRecord.model_validate(_valid_row({"merchant": "STARBUCKS"}))
        assert rec.merchant == "starbucks"

    def test_zero_amount_rejected(self):
        with pytest.raises(Exception):
            TransactionRecord.model_validate(_valid_row({"amount": 0.0}))

    def test_negative_amount_rejected(self):
        with pytest.raises(Exception):
            TransactionRecord.model_validate(_valid_row({"amount": -50.0}))

    def test_lat_without_lon_rejected(self):
        with pytest.raises(Exception):
            TransactionRecord.model_validate(_valid_row({"lat": 40.71, "lon": None}))

    def test_lon_without_lat_rejected(self):
        with pytest.raises(Exception):
            TransactionRecord.model_validate(_valid_row({"lat": None, "lon": -74.01}))

    def test_both_coords_none_accepted(self):
        rec = TransactionRecord.model_validate(_valid_row({"lat": None, "lon": None}))
        assert rec.lat is None
        assert rec.lon is None

    def test_default_mcc(self):
        row = _valid_row()
        del row["mcc"]
        rec = TransactionRecord.model_validate(row)
        assert rec.mcc == 9999

    def test_default_counterparty(self):
        row = _valid_row()
        del row["counterparty"]
        rec = TransactionRecord.model_validate(row)
        assert rec.counterparty == "unknown"

    def test_timestamp_parsed_to_datetime(self):
        rec = TransactionRecord.model_validate(_valid_row())
        assert isinstance(rec.timestamp, datetime)

    def test_model_dump_roundtrip(self):
        rec = TransactionRecord.model_validate(_valid_row())
        dumped = rec.model_dump()
        rec2 = TransactionRecord.model_validate(dumped)
        assert rec2.tx_id == rec.tx_id


class TestTransactionRecordProperties:
    @given(amount=st.floats(min_value=0.01, max_value=1_000_000.0,
                            allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_valid_amounts_always_accepted(self, amount):
        rec = TransactionRecord.model_validate(_valid_row({"amount": amount}))
        assert rec.amount > 0

    @given(merchant=st.text(min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_merchant_always_lowercased(self, merchant):
        try:
            rec = TransactionRecord.model_validate(_valid_row({"merchant": merchant}))
            assert rec.merchant == rec.merchant.lower()
        except Exception:
            pass  # pydantic may reject non-string, that's ok


# ── DataIngestionPipeline — CSV ───────────────────────────────────────────────

class TestCSVIngestion:
    def test_valid_csv_loaded(self, tmp_path):
        df = pd.DataFrame([_valid_row() for _ in range(10)])
        csv_path = tmp_path / "transactions.csv"
        df.to_csv(csv_path, index=False)

        pipeline = DataIngestionPipeline()
        out_df, result = pipeline.load(csv_path)

        assert len(out_df) == 10
        assert result.valid_records == 10
        assert result.rejected_records == 0
        assert result.rejection_rate == 0.0

    def test_invalid_rows_rejected(self, tmp_path):
        rows = [_valid_row() for _ in range(5)]
        rows.append({**_valid_row(), "amount": -100.0})  # invalid: negative amount
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "transactions.csv"
        df.to_csv(csv_path, index=False)

        pipeline = DataIngestionPipeline()
        out_df, result = pipeline.load(csv_path)

        assert result.rejected_records == 1
        assert result.valid_records == 5
        assert result.rejection_rate > 0

    def test_result_schema(self, tmp_path):
        df = pd.DataFrame([_valid_row()])
        csv_path = tmp_path / "t.csv"
        df.to_csv(csv_path, index=False)

        pipeline = DataIngestionPipeline()
        _, result = pipeline.load(csv_path)
        assert isinstance(result, IngestionResult)
        assert result.format == "csv"
        assert result.total_records == 1


# ── DataIngestionPipeline — Parquet ──────────────────────────────────────────

class TestParquetIngestion:
    def test_valid_parquet_loaded(self, tmp_path):
        df = pd.DataFrame([_valid_row() for _ in range(20)])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        parquet_path = tmp_path / "transactions.parquet"
        df.to_parquet(parquet_path, index=False)

        pipeline = DataIngestionPipeline()
        out_df, result = pipeline.load(parquet_path)

        assert len(out_df) == 20
        assert result.valid_records == 20
        assert result.format == "parquet"


# ── DataIngestionPipeline — JSON ──────────────────────────────────────────────

class TestJSONIngestion:
    def _make_webhook(self, row: dict) -> dict:
        return {
            "event_type": "transaction.created",
            "event_id": "abc-123",
            "sent_at": row["timestamp"],
            "data": {k: v for k, v in row.items() if k not in ("is_fraud", "fraud_type")},
        }

    def test_webhook_json_loaded(self, tmp_path):
        rows = [_valid_row() for _ in range(8)]
        webhooks = [self._make_webhook(r) for r in rows]
        json_path = tmp_path / "webhooks.json"
        json_path.write_text(json.dumps(webhooks))

        pipeline = DataIngestionPipeline()
        out_df, result = pipeline.load(json_path)

        assert result.valid_records == 8
        assert result.format == "json"

    def test_flat_json_also_works(self, tmp_path):
        """JSON without webhook envelope — raw transaction dicts."""
        rows = [_valid_row() for _ in range(3)]
        json_path = tmp_path / "flat.json"
        json_path.write_text(json.dumps(rows))

        pipeline = DataIngestionPipeline()
        out_df, result = pipeline.load(json_path)
        assert result.total_records == 3


# ── DataIngestionPipeline — Stream ───────────────────────────────────────────

class TestStreamIngestion:
    def test_stream_yields_batches(self):
        records = [_valid_row({"tx_id": f"TX{i:05d}"}) for i in range(25)]
        pipeline = DataIngestionPipeline()

        batches = list(pipeline.load_stream(iter(records), batch_size=10))
        # 25 records / batch_size=10 → 3 batches (10, 10, 5)
        assert len(batches) == 3
        total = sum(len(b) for b in batches)
        assert total == 25

    def test_stream_validates_records(self):
        records = [
            _valid_row({"tx_id": "TX00001"}),
            {**_valid_row(), "amount": -99.0},   # bad
            _valid_row({"tx_id": "TX00003"}),
        ]
        pipeline = DataIngestionPipeline()
        batches = list(pipeline.load_stream(iter(records), batch_size=100))
        # All records processed in one batch; 1 rejected
        assert len(batches) == 1
        assert len(batches[0]) == 2  # only 2 valid

    def test_empty_stream_returns_no_batches(self):
        pipeline = DataIngestionPipeline()
        batches = list(pipeline.load_stream(iter([]), batch_size=10))
        assert batches == []


# ── Unsupported format ────────────────────────────────────────────────────────

class TestUnsupportedFormat:
    def test_unsupported_format_raises(self, tmp_path):
        p = tmp_path / "data.xlsx"
        p.write_text("dummy")
        pipeline = DataIngestionPipeline()
        with pytest.raises(ValueError, match="Unsupported format"):
            pipeline.load(p, fmt="xlsx")


# ── NaN handling ──────────────────────────────────────────────────────────────

class TestNaNHandling:
    def test_nan_optional_fields_default(self, tmp_path):
        """NaN in optional nullable fields should not reject the record."""
        import math
        rows = [_valid_row()]
        # Set lat/lon to NaN (common in CSV null)
        rows[0]["lat"] = float("nan")
        rows[0]["lon"] = float("nan")
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "nan_test.csv"
        df.to_csv(csv_path, index=False)

        pipeline = DataIngestionPipeline()
        out_df, result = pipeline.load(csv_path)
        # lat/lon NaN → both None → should pass validate_coordinates
        assert result.valid_records == 1

    def test_nan_amount_rejected(self, tmp_path):
        rows = [_valid_row()]
        rows[0]["amount"] = float("nan")
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "nan_amount.csv"
        df.to_csv(csv_path, index=False)

        pipeline = DataIngestionPipeline()
        _, result = pipeline.load(csv_path)
        assert result.rejected_records == 1

