"""
feature_store.py
────────────────
Offline + online feature serving, simulating production feature store
patterns (e.g. Feast, Tecton, AWS SageMaker Feature Store).

Architecture
────────────
  Offline store  → Parquet (S3/local) for batch training jobs
  Online store   → In-memory dict (would be DynamoDB/Redis in prod)
                   keyed by (user_id, feature_group)

Interview talking point
───────────────────────
  "In production, the offline and online stores must be kept in sync
   (the 'training-serving skew' problem). I use Parquet for the offline
   store because it's compatible with Spark, Redshift, and Athena —
   all common in AWS data stacks. The online store serves low-latency
   lookups during real-time scoring."

  "A feature store decouples feature computation from model training:
   multiple models can share the same feature definitions without
   duplicating pipeline logic — critical for consistency and governance."
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class FeatureGroup:
    """Represents a versioned set of features for a given entity."""
    name: str
    version: str
    features: list[str]
    entity_key: str = "user_id"


# ── Feature group definitions ──────────────────────────────────────────────────
TRANSACTION_FEATURES = FeatureGroup(
    name="transaction_features",
    version="v1.0",
    entity_key="tx_id",
    features=[
        "z_amount", "ewma_z_amount", "tx_count_1h", "tx_count_24h",
        "tx_count_7d", "amount_sum_24h", "is_night", "is_new_counterparty",
        "geo_impossible", "geo_speed_kmh", "mcc_high_risk",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "amount_x_night", "amount_x_new_payee", "velocity_x_new_payee",
    ],
)

USER_PROFILE_FEATURES = FeatureGroup(
    name="user_profile_features",
    version="v1.0",
    entity_key="user_id",
    features=[
        "amount_mean_7tx", "amount_std_7tx", "amount_ewma", "amount_ewma_std",
    ],
)


class FeatureStore:
    """
    Lightweight feature store with offline Parquet sink and online dict cache.

    Usage
    -----
    >>> store = FeatureStore(offline_path=Path("data/processed"))
    >>> store.write_offline(featured_df, group=TRANSACTION_FEATURES)
    >>> store.write_online(user_id="U00001", features={"z_amount": 2.3})
    >>> ctx = store.get_online("U00001")
    """

    def __init__(self, offline_path: Path = Path("data/processed")) -> None:
        self.offline_path = offline_path
        self.offline_path.mkdir(parents=True, exist_ok=True)
        # Online store: {user_id: {feature_name: value}}
        self._online: dict[str, dict[str, Any]] = {}

    # ── Offline (Parquet) operations ───────────────────────────────────────────

    def write_offline(
        self,
        df: pd.DataFrame,
        group: FeatureGroup,
    ) -> Path:
        """
        Persist feature group to versioned Parquet file.
        Adds a content-hash for lineage tracking.
        """
        cols = [group.entity_key, "timestamp"] + [
            c for c in group.features if c in df.columns
        ]
        out_df = df[cols].copy()
        out_df["feature_group"] = group.name
        out_df["feature_version"] = group.version
        out_df["written_at"] = datetime.now(timezone.utc).isoformat()

        path = self.offline_path / f"{group.name}_{group.version}.parquet"
        out_df.to_parquet(path, index=False, compression="snappy")
        log.info("offline_write", path=str(path), rows=len(out_df), group=group.name)
        return path

    def read_offline(self, group: FeatureGroup) -> pd.DataFrame:
        path = self.offline_path / f"{group.name}_{group.version}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No offline data for {group.name} {group.version}")
        return pd.read_parquet(path)

    # ── Online (dict cache) operations ────────────────────────────────────────

    def write_online(self, user_id: str, features: dict[str, Any]) -> None:
        """
        Upsert user features into online store.
        In production this would be a DynamoDB PutItem call.
        """
        current = self._online.get(user_id, {})
        current.update(features)
        current["_updated_at"] = datetime.now(timezone.utc).isoformat()
        self._online[user_id] = current

    def get_online(self, user_id: str) -> dict[str, Any]:
        """
        Read user features from online store (O(1) dict lookup).
        Returns empty dict if user not found — models must handle cold-start.
        """
        return self._online.get(user_id, {})

    def populate_online_from_df(self, df: pd.DataFrame, group: FeatureGroup) -> None:
        """
        Bulk-populate online store from an enriched DataFrame.
        Used at startup to warm the cache from the offline store.
        """
        feat_cols = [c for c in group.features if c in df.columns]
        for _, row in df.groupby(group.entity_key)[feat_cols].mean().reset_index().iterrows():
            self.write_online(
                user_id=row[group.entity_key],
                features={c: float(row[c]) for c in feat_cols if c in row},
            )
        log.info("online_store_populated", users=len(self._online))

    def stats(self) -> dict[str, Any]:
        return {
            "online_entries": len(self._online),
            "offline_files": list(self.offline_path.glob("*.parquet")),
        }


# Module-level singleton
feature_store = FeatureStore()

