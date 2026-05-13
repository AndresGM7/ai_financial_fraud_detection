"""
enrichment.py
─────────────
Transforms raw validated transactions into behaviour-rich features.

Enrichment pipeline
───────────────────
  1. Merchant normalisation + MCC lookup
  2. Temporal features (hour, day-of-week, is_holiday, is_night)
  3. Rolling behavioural statistics per user (7-day window)
  4. Velocity features (tx count per 1h / 24h / 7d)
  5. Counterparty novelty (first-time payee flag)
  6. Geographic velocity (impossible travel detection)

Interview talking points
────────────────────────
  "I separate enrichment from feature engineering intentionally:
   enrichment adds domain-knowledge signals (merchant cleaning, velocity),
   while feature engineering creates model inputs (Z-scores, interaction
   terms). This separation makes each layer independently testable and
   allows the feature store to cache enriched data separately."

  "Rolling windows must be computed inside each user's group to avoid
   cross-user leakage — a subtlety that trips many candidates."

  "Geographic velocity: if user is in New York at 10:00 and Miami at
   10:45, the 1,280 km gap in 45 min is physically impossible. I flag
   that delta as a hard risk signal — it's one of the strongest ATO
   indicators in practice."
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.utils import get_logger, timed

log = get_logger(__name__)

# ── US Federal Holidays 2024 (simplified) ─────────────────────────────────────
_HOLIDAYS_2024: set[date] = {
    date(2024, 1, 1),   # New Year
    date(2024, 7, 4),   # Independence Day
    date(2024, 11, 28), # Thanksgiving
    date(2024, 12, 25), # Christmas
    date(2024, 1, 15),  # MLK Day
    date(2024, 2, 19),  # Presidents Day
    date(2024, 5, 27),  # Memorial Day
    date(2024, 9, 2),   # Labor Day
}

# ── MCC risk tiers ────────────────────────────────────────────────────────────
_HIGH_RISK_MCCS: set[int] = {6099, 6012, 6011, 7995, 5912}  # money transfer, ATM, gambling

# ── Merchant normaliser ───────────────────────────────────────────────────────
_MERCHANT_ALIASES: dict[str, str] = {
    "amzn": "amazon", "amazon.com": "amazon", "amazon": "amazon",
    "wmt": "walmart", "wal-mart": "walmart", "walmart": "walmart",
    "mcd": "mcdonalds", "mcdonald's": "mcdonalds", "mcdonalds": "mcdonalds",
    "starbucks coffee": "starbucks", "starbucks": "starbucks",
    "uber*": "uber", "uber": "uber",
    "lyft *": "lyft", "lyft": "lyft",
    "netflix.com": "netflix", "netflix": "netflix",
    "wgu": "western_union", "western union": "western_union",
    "western_union": "western_union",
}


def normalise_merchant(name: str) -> str:
    """
    Canonicalise raw merchant strings to a standard token.

    Why this matters in production:
      Raw merchant names from Visa/MC networks are notoriously messy:
      'AMZN*MKP 12345', 'AMAZON MKTP US', 'AMAZON.COM' all mean Amazon.
      Without normalisation, cardinality explodes and models can't learn
      merchant-level patterns.
    """
    if not isinstance(name, str) or not name.strip():
        return "unknown"
    cleaned = name.lower().strip()
    # exact alias lookup
    for alias, canonical in _MERCHANT_ALIASES.items():
        if alias in cleaned:
            return canonical
    # strip trailing digits / special chars (e.g. "UBER* TRIP 9AF2")
    cleaned = cleaned.split("*")[0].split("#")[0].strip()
    return cleaned


# ── Temporal features ─────────────────────────────────────────────────────────

def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek          # 0=Mon … 6=Sun
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = df["hour"].between(0, 5).astype(int)  # 00:00–05:59
    df["is_holiday"] = ts.dt.date.apply(lambda d: int(d in _HOLIDAYS_2024))
    # Cyclical encoding of hour (preserves 23→0 continuity for ML models)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


# ── Rolling behavioural stats ──────────────────────────────────────────────────

def _add_rolling_stats(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Rolling mean/std of amount per user over last `window` transactions.
    Computed inside user groups to prevent leakage across accounts.

    Interview talking point:
      "I use min_periods=1 so early transactions get a sensible estimate
       rather than NaN — NaN propagation silently zeros out Z-scores."
    """
    df = df.sort_values(["user_id", "timestamp"])
    roll = df.groupby("user_id")["amount"]

    df[f"amount_mean_{window}tx"] = roll.transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df[f"amount_std_{window}tx"] = roll.transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
    )
    # EWMA baseline (exponentially weighted — more weight to recent)
    df["amount_ewma"] = roll.transform(
        lambda x: x.shift(1).ewm(span=20, min_periods=1).mean()
    )
    df["amount_ewma_std"] = roll.transform(
        lambda x: x.shift(1).ewm(span=20, min_periods=1).std().fillna(0)
    )
    return df


# ── Velocity features ─────────────────────────────────────────────────────────

def _add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count transactions per user in rolling time windows.

    Note: pd.Grouper + rolling on DatetimeIndex gives true time-based windows,
    not row-count window — critical difference for irregular time series.
    """
    df = df.sort_values(["user_id", "timestamp"])
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True)

    results = []
    for uid, group in df.groupby("user_id"):
        g = group.set_index("timestamp_dt").sort_index()
        # rolling counts (shift(1) prevents look-ahead)
        g["tx_count_1h"] = (
            g["amount"].shift(1).rolling("1h", min_periods=0).count().fillna(0)
        )
        g["tx_count_24h"] = (
            g["amount"].shift(1).rolling("24h", min_periods=0).count().fillna(0)
        )
        g["tx_count_7d"] = (
            g["amount"].shift(1).rolling("7D", min_periods=0).count().fillna(0)
        )
        # rolling sum of amounts
        g["amount_sum_24h"] = (
            g["amount"].shift(1).rolling("24h", min_periods=0).sum().fillna(0)
        )
        g = g.reset_index(drop=True)
        results.append(g)

    enriched = pd.concat(results, ignore_index=True)
    df = df.drop(columns=["timestamp_dt"]).reset_index(drop=True)
    for col in ["tx_count_1h", "tx_count_24h", "tx_count_7d", "amount_sum_24h"]:
        df[col] = enriched[col].values
    return df


# ── Counterparty novelty ──────────────────────────────────────────────────────

def _add_counterparty_novelty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag transactions where the counterparty has not appeared in the
    user's last 30 transactions.

    Interview talking point:
      "New-counterparty flag is one of the highest-signal features for
       elder fraud: romance scammers and tech-support fraudsters almost
       always appear as brand-new payees."
    """
    df = df.sort_values(["user_id", "timestamp"])

    def is_new_counterparty(series: pd.Series) -> pd.Series:
        seen: set[str] = set()
        flags = []
        for val in series:
            flags.append(int(val not in seen))
            seen.add(val)
            if len(seen) > 30:
                # sliding window approximation
                seen = set(list(seen)[-30:])
        return pd.Series(flags, index=series.index)

    df["is_new_counterparty"] = (
        df.groupby("user_id")["counterparty"]
        .transform(is_new_counterparty)
    )
    return df


# ── Geographic velocity ───────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two GPS coordinates (Haversine formula).
    Returns kilometres.
    """
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _add_geo_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute km/h travel speed between consecutive transactions per user.
    Physically impossible speeds (>900 km/h, faster than commercial flight)
    are flagged as strong account-takeover signals.
    """
    df = df.sort_values(["user_id", "timestamp"]).copy()

    has_coords = df["lat"].notna() & df["lon"].notna()
    df["geo_speed_kmh"] = 0.0
    df["geo_impossible"] = 0

    for uid, group in df[has_coords].groupby("user_id"):
        if len(group) < 2:
            continue
        lats = group["lat"].values
        lons = group["lon"].values
        ts = pd.to_datetime(group["timestamp"], utc=True)
        hours = ts.diff().dt.total_seconds().fillna(0) / 3600

        speeds = [0.0]
        for i in range(1, len(group)):
            dist_km = _haversine_km(lats[i-1], lons[i-1], lats[i], lons[i])
            h = max(hours.iloc[i], 1 / 60)  # minimum 1 minute
            speeds.append(dist_km / h)

        df.loc[group.index, "geo_speed_kmh"] = speeds
        df.loc[group.index, "geo_impossible"] = [int(s > 900) for s in speeds]

    return df


# ── High-risk MCC flag ────────────────────────────────────────────────────────

def _add_mcc_risk(df: pd.DataFrame) -> pd.DataFrame:
    df["mcc_high_risk"] = df["mcc"].apply(lambda x: int(x in _HIGH_RISK_MCCS))
    return df


# ── Master enrichment function ────────────────────────────────────────────────

@timed
def enrich_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full enrichment pipeline in order.

    Parameters
    ----------
    df : validated DataFrame from ingestion layer

    Returns
    -------
    df_enriched : DataFrame with all behavioural + contextual features added
    """
    log.info("enrichment_start", rows=len(df))
    df = df.copy()

    # 1. Merchant normalisation
    df["merchant_clean"] = df["merchant"].apply(normalise_merchant)

    # 2. Temporal
    df = _add_temporal_features(df)

    # 3. Rolling behavioural stats
    df = _add_rolling_stats(df)

    # 4. Velocity
    df = _add_velocity_features(df)

    # 5. Counterparty novelty
    df = _add_counterparty_novelty(df)

    # 6. Geographic velocity
    df = _add_geo_velocity(df)

    # 7. MCC risk tier
    df = _add_mcc_risk(df)

    log.info("enrichment_complete", output_columns=len(df.columns))
    return df

