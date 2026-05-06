"""
Statistical detection models: Z-score rule, night rule, and EWMA-based anomaly detection.
"""
import numpy as np
import pandas as pd

from src.config import ZSCORE_THRESHOLD, NIGHT_AMOUNT_MULTIPLIER


def zscore_rule(df: pd.DataFrame, threshold: float = ZSCORE_THRESHOLD) -> pd.Series:
    """Flag transactions where |z_amount| exceeds threshold."""
    return (df["z_amount"].abs() > threshold).astype(int)


def night_rule(
    df: pd.DataFrame, multiplier: float = NIGHT_AMOUNT_MULTIPLIER
) -> pd.Series:
    """Flag nighttime transactions where amount is unusually high for the user."""
    return (
        (df["is_night"] == 1) & (df["amount"] > df["amount_mean_7"] * multiplier)
    ).astype(int)


def ewma_anomaly(df: pd.DataFrame, span: int = 10, sigma: float = 3.0) -> pd.Series:
    """
    EWMA-based anomaly: flag transactions where the deviation from the
    exponentially weighted moving average exceeds sigma * EWMA std.
    Computed per-user on sorted data.
    """
    df = df.copy()

    def _ewma_flag(series: pd.Series) -> pd.Series:
        ewma_mean = series.ewm(span=span, min_periods=1).mean()
        ewma_std = series.ewm(span=span, min_periods=1).std().fillna(0)
        deviation = (series - ewma_mean).abs()
        return (deviation > sigma * (ewma_std + 1e-6)).astype(int)

    return df.groupby("user_id")["amount"].transform(_ewma_flag)


def combine_statistical(df: pd.DataFrame) -> pd.Series:
    """Combine all statistical rules: flag if ANY rule triggers."""
    flags = (
        zscore_rule(df).values
        | night_rule(df).values
        | ewma_anomaly(df).values
    )
    return pd.Series(flags, index=df.index)
