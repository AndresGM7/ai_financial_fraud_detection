"""
ingestion.py
────────────
Multi-format transaction ingestion with Pydantic validation.

Supported sources
─────────────────
  • CSV   — batch historical files
  • Parquet — data lake / Redshift UNLOAD exports
  • JSON  — webhook event streams (real-time bank APIs)
  • Generator — Python iterator (Kafka/Kinesis simulation)

Interview talking points
────────────────────────
  "I built a format-agnostic ingestion layer behind a single interface.
   The Pydantic schema forces type coercion and rejects malformed records
   at the boundary — this is the 'fail fast' principle. In production,
   rejected records go to a dead-letter queue (SQS/DLQ) for manual review,
   not silently dropped."

  "Parquet is my preferred format for batch pipelines: columnar storage
   means I can read only the columns I need, cutting I/O by 80–90% vs CSV
   on wide tables — critical when transaction tables have 200+ columns."
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Iterator

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from src.utils import get_logger

log = get_logger(__name__)


# ── Pydantic schema ────────────────────────────────────────────────────────────

class TransactionRecord(BaseModel):
    """
    Canonical representation of a single financial transaction.
    All downstream modules consume this schema — never raw dicts.
    """
    tx_id: str
    user_id: str
    timestamp: datetime
    amount: float = Field(gt=0, description="Transaction amount in USD")
    merchant: str
    mcc: int = Field(default=9999, description="Merchant Category Code")
    tx_type: str = Field(description="ACH | ZELLE | WIRE | POS | ATM | ONLINE")
    counterparty: str = Field(default="unknown")
    currency: str = Field(default="USD")
    lat: float | None = None
    lon: float | None = None
    # Labels — present in labeled sets only, None in live data
    is_fraud: int | None = None
    fraud_type: str | None = None

    @field_validator("tx_type")
    @classmethod
    def normalise_tx_type(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("merchant")
    @classmethod
    def normalise_merchant_name(cls, v: str) -> str:
        return v.lower().strip() if isinstance(v, str) else "unknown"

    @field_validator("amount")
    @classmethod
    def round_amount(cls, v: float) -> float:
        return round(v, 2)

    @model_validator(mode="after")
    def validate_coordinates(self) -> "TransactionRecord":
        if (self.lat is None) != (self.lon is None):
            raise ValueError("lat and lon must both be present or both absent")
        return self


class IngestionResult(BaseModel):
    """Summary of a single ingestion run."""
    source: str
    format: str
    total_records: int
    valid_records: int
    rejected_records: int
    rejection_rate: float
    rejected_samples: list[dict[str, Any]] = Field(default_factory=list)


# ── Ingestion pipeline ─────────────────────────────────────────────────────────

class DataIngestionPipeline:
    """
    Unified ingestion interface.

    Usage
    -----
    >>> pipeline = DataIngestionPipeline()
    >>> df, result = pipeline.load("data/raw/transactions.csv")
    >>> df, result = pipeline.load("data/raw/transactions.parquet")
    >>> df, result = pipeline.load("data/raw/webhooks.json")
    """

    def load(
        self,
        source: str | Path,
        fmt: str | None = None,
    ) -> tuple[pd.DataFrame, IngestionResult]:
        """
        Auto-detect format from extension or use explicit fmt argument.
        Returns (validated DataFrame, IngestionResult summary).
        """
        path = Path(source)
        fmt = fmt or path.suffix.lstrip(".").lower()

        loaders = {
            "csv": self._load_csv,
            "parquet": self._load_parquet,
            "json": self._load_json_webhooks,
        }
        if fmt not in loaders:
            raise ValueError(f"Unsupported format '{fmt}'. Choose from: {list(loaders)}")

        raw_df = loaders[fmt](path)
        return self._validate(raw_df, source=str(path), fmt=fmt)

    def load_stream(
        self,
        stream: Iterator[dict[str, Any]],
        batch_size: int = 100,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Consume a Python iterator (simulates Kinesis/Kafka stream).
        Yields validated DataFrames in batches.

        Interview talking point:
          "In production we'd connect this to a Kinesis consumer or a
           Kafka consumer group — the batch_size controls memory pressure
           vs. throughput tradeoff."
        """
        batch: list[dict] = []
        for record in stream:
            batch.append(record)
            if len(batch) >= batch_size:
                df_raw = pd.DataFrame(batch)
                df_valid, _ = self._validate(df_raw, source="stream", fmt="stream")
                yield df_valid
                batch = []
        if batch:
            df_raw = pd.DataFrame(batch)
            df_valid, _ = self._validate(df_raw, source="stream", fmt="stream")
            yield df_valid

    # ── Format-specific loaders ────────────────────────────────────────────────

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        log.info("csv_loaded", path=str(path), rows=len(df))
        return df

    @staticmethod
    def _load_parquet(path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        log.info("parquet_loaded", path=str(path), rows=len(df))
        return df

    @staticmethod
    def _load_json_webhooks(path: Path) -> pd.DataFrame:
        """
        Parse webhook-style JSON array.
        Each event: {"event_type": ..., "data": {...}}
        """
        events = json.loads(path.read_text())
        rows = []
        for event in events:
            row = event.get("data", event)  # unwrap envelope if present
            rows.append(row)
        df = pd.DataFrame(rows)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        log.info("json_webhooks_loaded", path=str(path), events=len(events))
        return df

    # ── Pydantic validation pass ───────────────────────────────────────────────

    @staticmethod
    def _validate(
        raw_df: pd.DataFrame,
        source: str,
        fmt: str,
    ) -> tuple[pd.DataFrame, IngestionResult]:
        """
        Validate every row against TransactionRecord.
        Invalid rows go to dead-letter log — never silently dropped in prod.
        """
        valid_rows: list[dict] = []
        rejected: list[dict] = []

        import math
        for row in raw_df.to_dict(orient="records"):
            # Replace float NaN (from CSV nulls) with None before Pydantic
            row = {
                k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in row.items()
            }
            try:
                record = TransactionRecord.model_validate(row)
                valid_rows.append(record.model_dump())
            except Exception as exc:
                rejected.append({"row": row, "error": str(exc)})

        if rejected:
            log.warning(
                "records_rejected",
                count=len(rejected),
                examples=[r["error"] for r in rejected[:3]],
            )

        valid_df = pd.DataFrame(valid_rows) if valid_rows else pd.DataFrame()
        result = IngestionResult(
            source=source,
            format=fmt,
            total_records=len(raw_df),
            valid_records=len(valid_rows),
            rejected_records=len(rejected),
            rejection_rate=round(len(rejected) / max(len(raw_df), 1), 4),
            rejected_samples=rejected[:5],
        )
        log.info(
            "ingestion_complete",
            valid=result.valid_records,
            rejected=result.rejected_records,
            rejection_rate=result.rejection_rate,
        )
        return valid_df, result

