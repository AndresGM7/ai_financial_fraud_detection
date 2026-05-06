"""
Data enrichment: merchant normalization, temporal features, and rolling statistics.
"""
import pandas as pd
import numpy as np


_MERCHANT_MAP = {
    "amazon": "amazon",
    "amzn": "amazon",
    "walmart": "walmart",
    "wmt": "walmart",
    "target": "target",
    "tgt": "target",
    "netflix": "netflix",
    "spotify": "spotify",
    "apple": "apple",
    "google": "google",
    "paypal": "paypal",
    "uber": "uber",
    "lyft": "lyft",
    "airbnb": "airbnb",
}


def normalize_merchant(name: str) -> str:
    """Normalize merchant name to reduce cardinality."""
    if not isinstance(name, str):
        return "unknown"
    name_lower = name.lower().strip()
    for key, canonical in _MERCHANT_MAP.items():
        if key in name_lower:
            return canonical
    return name_lower


def enrich_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich raw transactions with:
    - Normalized merchant names
    - Temporal features (hour, day_of_week, is_night, is_weekend)
    - Per-user rolling statistics (mean and std of amount over last 7 transactions)
    """
    df = df.copy()
    df["merchant_clean"] = df["merchant"].apply(normalize_merchant)

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Per-user rolling statistics (sort first to ensure time ordering)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["amount_mean_7"] = (
        df.groupby("user_id")["amount"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    df["amount_std_7"] = (
        df.groupby("user_id")["amount"]
        .transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))
    )

    return df
