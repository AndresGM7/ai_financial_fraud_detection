"""
Feature engineering: derive risk-proxy features from enriched transactions.
"""
import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build risk features from an enriched DataFrame.

    Features:
    - z_amount: z-score of transaction amount relative to user's recent history
    - is_new_merchant: flag if this merchant hasn't appeared in the user's last 20 txs
    - tx_count_24h: number of transactions by user in the last 24 hours
    - amount_ratio: transaction amount relative to the user's rolling mean
    """
    df = df.copy()

    # Z-score of amount relative to user's 7-tx rolling mean/std
    df["z_amount"] = (df["amount"] - df["amount_mean_7"]) / (
        df["amount_std_7"] + 1e-6
    )

    # Flag transactions where merchant is "new" (not seen in last 20 txs per user)
    def _is_new_merchant(series: pd.Series) -> pd.Series:
        seen: set = set()
        result = []
        for merchant in series:
            result.append(int(merchant not in seen))
            seen.add(merchant)
            if len(seen) > 20:
                # keep only the most recent 20 unique merchants
                seen = set(list(seen)[-20:])
        return pd.Series(result, index=series.index)

    df["is_new_merchant"] = (
        df.groupby("user_id")["merchant_clean"]
        .transform(_is_new_merchant)
        .astype(int)
    )

    # Amount ratio to rolling mean (clipped to avoid infinity)
    df["amount_ratio"] = (df["amount"] / (df["amount_mean_7"] + 1e-6)).clip(0, 100)

    # Transaction count in the last 24 hours per user
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    tx_count = pd.Series(0, index=df.index, dtype=int)
    for _, group in df.groupby("user_id"):
        ts = group["timestamp"]
        counts = []
        ts_list = ts.tolist()
        for i, t in enumerate(ts_list):
            counts.append(int(sum(1 for s in ts_list[: i + 1] if (t - s) <= pd.Timedelta(hours=24))))
        tx_count.loc[group.index] = counts

    df["tx_count_24h"] = tx_count

    return df


def get_feature_columns() -> list:
    """Return the list of numeric feature columns used by ML models."""
    return [
        "amount",
        "hour",
        "day_of_week",
        "is_night",
        "is_weekend",
        "amount_mean_7",
        "amount_std_7",
        "z_amount",
        "is_new_merchant",
        "amount_ratio",
        "tx_count_24h",
    ]
