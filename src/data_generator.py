"""
data_generator.py
─────────────────
Generates realistic synthetic financial transaction data with injected
elder-fraud patterns drawn from CFPB and AARP research typologies.

Outputs
───────
  • data/raw/transactions.csv      – flat CSV (classic tabular)
  • data/raw/transactions.parquet  – columnar (Parquet, production-grade)
  • data/raw/webhooks.json         – list of JSON webhook payloads
                                     (simulates bank real-time event stream)
  • data/raw/labeled_transactions.csv – same set with fraud labels for
                                     supervised learning experiments

Interview talking points
────────────────────────
  "I injected five distinct elder-fraud typologies:
   1. Romance/tech-support scam  — unusually large first-time wire transfer
   2. Lottery/prize scam         — rapid small outbound ACH to new payee
   3. Card-not-present fraud     — online purchases at 3–5 AM across new merchants
   4. Account-takeover burst     — >5 transactions in 60 minutes to new payees
   5. Structuring               — multiple sub-$10k cash withdrawals to stay
                                   below BSA reporting threshold

  Each pattern has configurable probability so I can tune the synthetic
  fraud rate independently of the feature-engineering decisions."
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker

from src.config import settings

fake = Faker()
rng = np.random.default_rng(settings.random_seed)
random.seed(settings.random_seed)
Faker.seed(settings.random_seed)

# ── Merchant Category Codes (MCC) — simplified lookup ─────────────────────────
MCC_MAP: dict[str, int] = {
    "amazon": 5999,
    "walmart": 5411,
    "target": 5311,
    "cvs": 5912,
    "walgreens": 5912,
    "whole_foods": 5411,
    "starbucks": 5812,
    "mcdonalds": 5812,
    "uber": 4121,
    "lyft": 4121,
    "netflix": 7922,
    "apple": 5732,
    "google": 7372,
    "western_union": 6099,   # money transfer — high-risk
    "moneygram": 6099,
    "wire_transfer": 6012,
    "atm_withdrawal": 6011,
    "unknown": 9999,
}

NORMAL_MERCHANTS = [
    "amazon", "walmart", "target", "cvs", "walgreens",
    "whole_foods", "starbucks", "mcdonalds", "uber", "lyft",
    "netflix", "apple",
]

HIGH_RISK_MERCHANTS = ["western_union", "moneygram", "wire_transfer"]

TX_TYPES = ["ACH", "ZELLE", "WIRE", "POS", "ATM", "ONLINE"]

US_CITIES = [
    ("New York", "NY", 40.71, -74.01),
    ("Los Angeles", "CA", 34.05, -118.24),
    ("Chicago", "IL", 41.88, -87.63),
    ("Houston", "TX", 29.76, -95.37),
    ("Miami", "FL", 25.77, -80.19),
    ("Seattle", "WA", 47.61, -122.33),
]


# ── User profile generation ───────────────────────────────────────────────────

def _make_user(user_id: int) -> dict[str, Any]:
    city = random.choice(US_CITIES)
    age = rng.integers(62, 90)  # older-adult demographic (Carefull's focus)
    return {
        "user_id": f"U{user_id:05d}",
        "name": fake.name(),
        "age": int(age),
        "city": city[0],
        "state": city[1],
        "home_lat": city[2] + rng.uniform(-0.5, 0.5),
        "home_lon": city[3] + rng.uniform(-0.5, 0.5),
        "avg_monthly_spend": float(rng.uniform(500, 5000)),
        "typical_merchants": random.sample(NORMAL_MERCHANTS, k=5),
    }


# ── Normal transaction factory ────────────────────────────────────────────────

def _normal_tx(
    tx_id: int,
    user: dict,
    base_dt: datetime,
) -> dict[str, Any]:
    merchant = random.choice(user["typical_merchants"])
    amount = float(
        rng.lognormal(
            mean=np.log(user["avg_monthly_spend"] / 30),
            sigma=0.6,
        )
    )
    amount = round(min(amount, 4999.0), 2)
    hour = int(rng.choice(range(8, 22), p=None))  # daytime
    dt = base_dt.replace(hour=hour, minute=int(rng.integers(0, 59)))
    lat = user["home_lat"] + rng.uniform(-0.1, 0.1)
    lon = user["home_lon"] + rng.uniform(-0.1, 0.1)

    return {
        "tx_id": f"TX{tx_id:08d}",
        "user_id": user["user_id"],
        "timestamp": dt.isoformat(),
        "amount": amount,
        "merchant": merchant,
        "mcc": MCC_MAP.get(merchant, 9999),
        "tx_type": random.choice(["POS", "ONLINE", "ACH"]),
        "counterparty": fake.company(),
        "currency": "USD",
        "lat": round(float(lat), 4),
        "lon": round(float(lon), 4),
        "is_fraud": 0,
        "fraud_type": None,
    }


# ── Fraud pattern factories ───────────────────────────────────────────────────

def _romance_scam_tx(tx_id: int, user: dict, base_dt: datetime) -> dict[str, Any]:
    """Large first-time wire to unknown foreign entity."""
    tx = _normal_tx(tx_id, user, base_dt)
    tx.update({
        "tx_id": f"TX{tx_id:08d}",
        "amount": round(float(rng.uniform(3000, 25000)), 2),
        "merchant": "wire_transfer",
        "mcc": MCC_MAP["wire_transfer"],
        "tx_type": "WIRE",
        "counterparty": fake.name() + " (overseas)",
        "is_fraud": 1,
        "fraud_type": "romance_scam",
    })
    return tx


def _lottery_scam_tx(tx_id: int, user: dict, base_dt: datetime) -> list[dict]:
    """Rapid small ACH to money-transfer service (3–8 payments within 2 hours)."""
    txs = []
    for i in range(int(rng.integers(3, 8))):
        tx = _normal_tx(tx_id + i, user, base_dt)
        minute_offset = i * int(rng.integers(10, 20))
        dt = base_dt + timedelta(minutes=minute_offset)
        tx.update({
            "tx_id": f"TX{(tx_id + i):08d}",
            "timestamp": dt.isoformat(),
            "amount": round(float(rng.uniform(200, 900)), 2),
            "merchant": random.choice(HIGH_RISK_MERCHANTS),
            "mcc": MCC_MAP["western_union"],
            "tx_type": "ACH",
            "counterparty": "PRIZE CLAIM CENTER",
            "is_fraud": 1,
            "fraud_type": "lottery_scam",
        })
        txs.append(tx)
    return txs


def _card_not_present_tx(tx_id: int, user: dict, base_dt: datetime) -> dict[str, Any]:
    """3–5 AM online purchase at unfamiliar merchant."""
    tx = _normal_tx(tx_id, user, base_dt)
    dt = base_dt.replace(hour=int(rng.integers(1, 5)), minute=int(rng.integers(0, 59)))
    tx.update({
        "tx_id": f"TX{tx_id:08d}",
        "timestamp": dt.isoformat(),
        "amount": round(float(rng.uniform(100, 1500)), 2),
        "merchant": fake.domain_word(),   # unfamiliar merchant
        "mcc": 5999,
        "tx_type": "ONLINE",
        "counterparty": fake.company(),
        "lat": round(float(rng.uniform(25, 48)), 4),
        "lon": round(float(rng.uniform(-120, -70)), 4),
        "is_fraud": 1,
        "fraud_type": "card_not_present",
    })
    return tx


def _account_takeover_burst(tx_id: int, user: dict, base_dt: datetime) -> list[dict]:
    """> 5 tx in 60 min to different new payees."""
    txs = []
    for i in range(int(rng.integers(5, 9))):
        tx = _normal_tx(tx_id + i, user, base_dt)
        dt = base_dt + timedelta(minutes=i * int(rng.integers(5, 12)))
        tx.update({
            "tx_id": f"TX{(tx_id + i):08d}",
            "timestamp": dt.isoformat(),
            "amount": round(float(rng.uniform(50, 500)), 2),
            "merchant": fake.company().lower().replace(" ", "_"),
            "tx_type": "ZELLE",
            "counterparty": fake.name(),
            "is_fraud": 1,
            "fraud_type": "account_takeover",
        })
        txs.append(tx)
    return txs


def _structuring_tx(tx_id: int, user: dict, base_dt: datetime) -> list[dict]:
    """Several withdrawals just under $10,000 (BSA threshold)."""
    txs = []
    for i in range(int(rng.integers(2, 5))):
        tx = _normal_tx(tx_id + i, user, base_dt)
        dt = base_dt + timedelta(hours=i * int(rng.integers(1, 6)))
        amount = round(float(rng.uniform(8000, 9800)), 2)
        tx.update({
            "tx_id": f"TX{(tx_id + i):08d}",
            "timestamp": dt.isoformat(),
            "amount": amount,
            "merchant": "atm_withdrawal",
            "mcc": MCC_MAP["atm_withdrawal"],
            "tx_type": "ATM",
            "counterparty": "CASH",
            "is_fraud": 1,
            "fraud_type": "structuring",
        })
        txs.append(tx)
    return txs


FRAUD_FACTORIES = [
    _romance_scam_tx,
    _card_not_present_tx,
]

FRAUD_BURST_FACTORIES = [
    _lottery_scam_tx,
    _account_takeover_burst,
    _structuring_tx,
]


# ── Main generator ────────────────────────────────────────────────────────────

def generate_transactions(
    n_users: int | None = None,
    n_transactions: int | None = None,
    fraud_rate: float | None = None,
) -> pd.DataFrame:
    """
    Generate a DataFrame of synthetic transactions.

    Parameters
    ----------
    n_users       : number of unique users (defaults to settings)
    n_transactions: total transactions to generate
    fraud_rate    : fraction of fraud transactions

    Returns
    -------
    pd.DataFrame with columns matching TransactionRecord schema
    """
    n_users = n_users or settings.synthetic_n_users
    n_transactions = n_transactions or settings.synthetic_n_transactions
    fraud_rate = fraud_rate or settings.synthetic_fraud_rate

    users = [_make_user(i) for i in range(n_users)]
    records: list[dict] = []
    tx_id = 0

    start_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)

    for _ in range(n_transactions):
        user = random.choice(users)
        days_offset = int(rng.integers(0, 365))
        base_dt = start_dt + timedelta(days=days_offset)

        roll = rng.random()
        if roll < fraud_rate * 0.4:
            # Single-transaction fraud
            factory = random.choice(FRAUD_FACTORIES)
            records.append(factory(tx_id, user, base_dt))
            tx_id += 1
        elif roll < fraud_rate:
            # Multi-transaction fraud burst
            factory = random.choice(FRAUD_BURST_FACTORIES)
            burst = factory(tx_id, user, base_dt)
            records.extend(burst)
            tx_id += len(burst)
        else:
            records.append(_normal_tx(tx_id, user, base_dt))
            tx_id += 1

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


# ── Export helpers ────────────────────────────────────────────────────────────

def save_all_formats(df: pd.DataFrame, out_dir: Path = Path("data/raw")) -> None:
    """
    Persist transactions in three formats:
      1. CSV   — universally compatible, easy for quick exploration
      2. Parquet — columnar, compressed, used in production data lakes (S3/Redshift)
      3. JSON  — webhook-style event stream, mirrors real-time bank API payloads
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. CSV
    csv_path = out_dir / "transactions.csv"
    df.to_csv(csv_path, index=False)
    print(f"[✓] CSV saved: {csv_path}  ({len(df):,} rows)")

    # 2. Parquet (columnar, ~5× smaller than CSV, O(1) column access)
    parquet_path = out_dir / "transactions.parquet"
    df.to_parquet(parquet_path, index=False, compression="snappy")
    print(f"[✓] Parquet saved: {parquet_path}")

    # 3. JSON — webhook event envelope mimicking a bank real-time feed
    webhook_path = out_dir / "webhooks.json"
    webhooks = []
    for _, row in df.iterrows():
        payload = {
            "event_type": "transaction.created",
            "event_id": str(fake.uuid4()),
            "sent_at": row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"]),
            "data": {
                k: (v.isoformat() if hasattr(v, "isoformat") else v)
                for k, v in row.items()
                if k not in ("is_fraud", "fraud_type")  # labels stripped in live feed
            },
        }
        webhooks.append(payload)
    webhook_path.write_text(json.dumps(webhooks, indent=2, default=str))
    print(f"[✓] Webhooks JSON saved: {webhook_path}  ({len(webhooks):,} events)")

    # 4. Labeled CSV for supervised learning experiments
    labeled_path = out_dir / "labeled_transactions.csv"
    df.to_csv(labeled_path, index=False)
    n_fraud = df["is_fraud"].sum()
    print(
        f"[✓] Labeled CSV saved: {labeled_path}  "
        f"(fraud={n_fraud:,}, {n_fraud/len(df)*100:.2f}%)"
    )


if __name__ == "__main__":
    print("Generating synthetic transaction dataset…")
    df = generate_transactions()
    save_all_formats(df)
    print(f"\nDataset shape: {df.shape}")
    print(df["fraud_type"].value_counts(dropna=False))

