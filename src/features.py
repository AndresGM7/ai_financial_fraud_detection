"""
features.py
───────────
Transforms enriched transactions into model-ready feature vectors.

Feature families
────────────────
  • Deviation features  (Z-scores, EWMA-normalised)
  • Novelty features    (new counterparty, new merchant category)
  • Velocity features   (already computed in enrichment, encoded here)
  • Interaction features(amount × is_night, amount × is_new_counterparty)
  • One-hot encoding    (tx_type)

Interview talking points
────────────────────────
  "I encode deviations, not raw values. A $5,000 transaction means nothing
   without knowing the user's baseline — a Z-score of 0.5 versus 8.0 is
   the actual signal."

  "I add interaction terms manually because tree models (XGBoost) discover
   them, but linear models (LogReg) and anomaly detectors don't — being
   explicit avoids a common model-class gotcha."

  "I scale features only for models that are sensitive to magnitude
   (IsolationForest, LOF, Autoencoder, LogReg). XGBoost is scale-invariant,
   so I keep a raw copy as well."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger

log = get_logger(__name__)

# ── Feature column registry ────────────────────────────────────────────────────
# Explicit lists make feature selection transparent and reproducible.

NUMERICAL_FEATURES = [
    "amount",
    "z_amount",           # Z-score vs user's 7-tx rolling mean
    "ewma_z_amount",      # EWMA-normalised deviation
    "amount_mean_7tx",
    "amount_std_7tx",
    "tx_count_1h",
    "tx_count_24h",
    "tx_count_7d",
    "amount_sum_24h",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "geo_speed_kmh",
]

BINARY_FEATURES = [
    "is_night",
    "is_weekend",
    "is_holiday",
    "is_new_counterparty",
    "geo_impossible",
    "mcc_high_risk",
]

INTERACTION_FEATURES = [
    "amount_x_night",         # large-amount + night hours
    "amount_x_new_payee",     # large-amount + new counterparty
    "velocity_x_new_payee",   # high-velocity + new counterparty
]

TX_TYPE_DUMMIES = ["tx_type_ACH", "tx_type_WIRE", "tx_type_ZELLE",
                   "tx_type_ATM", "tx_type_ONLINE", "tx_type_POS"]

ALL_FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES + INTERACTION_FEATURES + TX_TYPE_DUMMIES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from enriched data.
    Returns a copy with new feature columns appended.
    """
    log.info("feature_engineering_start", rows=len(df))
    df = df.copy()

    # ── 1. Z-score vs user rolling baseline ───────────────────────────────────
    # Z = (x - μ) / (σ + ε)   ε prevents division by zero for new users
    eps = 1e-6
    df["z_amount"] = (
        (df["amount"] - df["amount_mean_7tx"]) /
        (df["amount_std_7tx"] + eps)
    )

    # ── 2. EWMA-normalised deviation ──────────────────────────────────────────
    df["ewma_z_amount"] = (
        (df["amount"] - df["amount_ewma"]) /
        (df["amount_ewma_std"] + eps)
    )

    # ── 3. Interaction features ───────────────────────────────────────────────
    df["amount_x_night"] = df["amount"] * df["is_night"]
    df["amount_x_new_payee"] = df["amount"] * df["is_new_counterparty"]
    df["velocity_x_new_payee"] = df["tx_count_1h"] * df["is_new_counterparty"]

    # ── 4. One-hot encode transaction type ────────────────────────────────────
    tx_dummies = pd.get_dummies(df["tx_type"], prefix="tx_type")
    for col in TX_TYPE_DUMMIES:
        dummy_col = tx_dummies.get(col)
        if dummy_col is not None:
            df[col] = dummy_col.astype(int)
        else:
            df[col] = 0

    # ── 5. Clip extreme Z-scores (winsorize at ±10) ───────────────────────────
    # Outlier transactions shouldn't dominate linear models
    df["z_amount"] = df["z_amount"].clip(-10, 10)
    df["ewma_z_amount"] = df["ewma_z_amount"].clip(-10, 10)

    # ── 6. Fill any remaining NaNs ────────────────────────────────────────────
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    log.info("feature_engineering_complete", n_features=len(ALL_FEATURES))
    return df


def get_feature_matrix(
    df: pd.DataFrame,
    scale: bool = False,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, StandardScaler | None]:
    """
    Extract the numeric feature matrix from the featured DataFrame.

    Parameters
    ----------
    df     : output of build_features()
    scale  : whether to apply StandardScaler (needed for LOF, LogReg, Autoencoder)
    scaler : pre-fitted scaler for inference; if None and scale=True, fit a new one

    Returns
    -------
    (X, scaler)  — X is np.ndarray, scaler is None if scale=False
    """
    available = [c for c in ALL_FEATURES if c in df.columns]
    missing = set(ALL_FEATURES) - set(available)
    if missing:
        log.warning("missing_features", columns=list(missing))

    X = df[available].fillna(0).values.astype(np.float32)

    if scale:
        if scaler is None:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)

    return X, scaler


def get_labels(df: pd.DataFrame) -> np.ndarray | None:
    """Return label array if 'is_fraud' column is present, else None."""
    if "is_fraud" not in df.columns or df["is_fraud"].isna().all():
        return None
    return df["is_fraud"].fillna(0).values.astype(int)

