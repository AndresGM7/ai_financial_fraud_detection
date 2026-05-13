"""
models/statistical.py
─────────────────────
Rule-based and statistical anomaly detectors.

Detectors implemented
─────────────────────
  1. Z-score rule         — flags when a transaction deviates > N std devs
                            from the user's rolling mean.
  2. EWMA rule            — same but using an exponentially-weighted baseline;
                            adapts faster to legitimate spending shifts.
  3. CUSUM               — cumulative sum control chart; catches gradual drift
                            that single-point detectors miss.
  4. Rules engine         — YAML-configurable hard rules (night + wire + new payee)
  5. Ensemble             — weighted OR of all detectors

Interview talking points
────────────────────────
  "Statistical rules are my baseline — they are fast (microseconds), fully
   interpretable, and require no labels. I use them as a first filter and
   to generate the 'statistical_score' signal that feeds the decision engine.

  CUSUM is worth explaining in depth: unlike Z-score which looks at a single
  point, CUSUM accumulates deviations over time. It detects when a user's
  spending gradually shifts upward — exactly the pattern in a romance scam
  where the fraudster slowly escalates wire transfer amounts over weeks."

  Z-score assumption: amounts are log-normally distributed (they are in practice
  — most retail purchases cluster around a mean but with fat right tail).
  I log-transform before computing Z-scores for better calibration."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import settings
from src.utils import get_logger

log = get_logger(__name__)


# ── 1. Z-score detector ───────────────────────────────────────────────────────

def zscore_flag(
    df: pd.DataFrame,
    threshold: float | None = None,
    use_log: bool = True,
) -> pd.Series:
    """
    Flag transactions where |Z-score| exceeds threshold.

    Parameters
    ----------
    df        : enriched DataFrame (must have z_amount column)
    threshold : std-dev multiplier (default from settings)
    use_log   : if True, applies log-amount Z-score for heavy-tailed dist

    Returns
    -------
    pd.Series of {0, 1} flags
    """
    threshold = threshold or settings.zscore_threshold

    if use_log:
        # Log-transform to handle lognormal distribution of spending amounts
        log_amount = np.log1p(df["amount"])
        mu = df["amount_mean_7tx"].apply(np.log1p)
        sigma = df["amount_std_7tx"] + 1e-6
        z = (log_amount - mu) / sigma
    else:
        z = df["z_amount"]

    flag = (z.abs() > threshold).astype(int)
    log.debug("zscore_flags", flagged=int(flag.sum()), threshold=threshold)
    return flag


def zscore_score(df: pd.DataFrame) -> pd.Series:
    """Return a continuous risk score [0, 1] from Z-scores (sigmoid-normalised)."""
    z = df["z_amount"].abs()
    return (1 / (1 + np.exp(-(z - 3)))).clip(0, 1)


# ── 2. EWMA detector ──────────────────────────────────────────────────────────

def ewma_flag(df: pd.DataFrame, threshold: float | None = None) -> pd.Series:
    """
    Flag transactions deviating significantly from EWMA baseline.
    EWMA adapts to recent behaviour — better than static rolling mean
    when spending patterns change (e.g. holiday season).
    """
    threshold = threshold or settings.zscore_threshold
    z_ewma = df["ewma_z_amount"].abs()
    return (z_ewma > threshold).astype(int)


def ewma_score(df: pd.DataFrame) -> pd.Series:
    return (1 / (1 + np.exp(-(df["ewma_z_amount"].abs() - 3)))).clip(0, 1)


# ── 3. CUSUM detector ─────────────────────────────────────────────────────────

def cusum_scores(
    series: pd.Series,
    target_mean: float | None = None,
    sigma: float | None = None,
    drift_threshold: float | None = None,
) -> pd.Series:
    """
    CUSUM (Cumulative Sum) control chart for detecting persistent upward drift.

    Algorithm
    ---------
    S_t = max(0, S_{t-1} + (x_t - μ - k))
    where k = σ/2 (slack parameter)
    Alert when S_t > h (drift_threshold)

    Interview talking point
    -----------------------
    "CUSUM accumulates evidence of a shift: if a user's amounts are 20%
     above average for 10 consecutive days, CUSUM catches it; a Z-score
     test on any single transaction would not."
    """
    drift_threshold = drift_threshold or settings.cusum_drift_threshold
    values = series.values.astype(float)
    mu = target_mean if target_mean is not None else np.mean(values[:10])
    s = sigma if sigma is not None else (np.std(values[:10]) + 1e-6)
    k = s / 2  # slack (half sigma)

    cusum_pos = np.zeros(len(values))
    for i in range(1, len(values)):
        cusum_pos[i] = max(0.0, cusum_pos[i-1] + (values[i] - mu - k))

    # Normalise to [0, 1]
    max_val = float(cusum_pos.max()) if cusum_pos.max() > 0 else 1.0
    return pd.Series(cusum_pos / max_val, index=series.index)


def cusum_flag_per_user(
    df: pd.DataFrame,
    drift_threshold: float | None = None,
) -> pd.Series:
    """Apply CUSUM per user and return binary flags."""
    drift_threshold = drift_threshold or settings.cusum_drift_threshold
    flags = pd.Series(0, index=df.index)
    for uid, group in df.groupby("user_id"):
        scores = cusum_scores(
            group["amount"],
            drift_threshold=drift_threshold,
        )
        flags.loc[group.index] = (scores > 0.8).astype(int)
    return flags


# ── 4. Rules engine ───────────────────────────────────────────────────────────

@dataclass
class Rule:
    name: str
    description: str
    weight: float = 1.0

    def apply(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


@dataclass
class NightLargeWireRule(Rule):
    """Large wire transfer during night hours to a new counterparty."""
    name: str = "night_large_wire"
    description: str = "Wire > 3× user baseline at night to new counterparty"
    weight: float = 0.9

    def apply(self, df: pd.DataFrame) -> pd.Series:
        return (
            (df["is_night"] == 1) &
            (df.get("tx_type_WIRE", pd.Series(0, index=df.index)) == 1) &
            (df["amount"] > df["amount_mean_7tx"] * 3) &
            (df["is_new_counterparty"] == 1)
        ).astype(int)


@dataclass
class RapidSuccessionRule(Rule):
    """Multiple transactions within 1 hour (account takeover signal)."""
    name: str = "rapid_succession"
    description: str = "> 5 transactions in last hour"
    weight: float = 0.8

    def apply(self, df: pd.DataFrame) -> pd.Series:
        return (df["tx_count_1h"] > 5).astype(int)


@dataclass
class GeographicImpossibilityRule(Rule):
    """Physically impossible travel speed between consecutive transactions."""
    name: str = "geo_impossible"
    description: str = "Travel speed > 900 km/h between transactions"
    weight: float = 0.95

    def apply(self, df: pd.DataFrame) -> pd.Series:
        return df.get("geo_impossible", pd.Series(0, index=df.index)).astype(int)


@dataclass
class StructuringRule(Rule):
    """Multiple ATM withdrawals just under $10,000 (BSA threshold)."""
    name: str = "structuring"
    description: str = "ATM withdrawal $8,000–$9,999 repeated"
    weight: float = 0.85

    def apply(self, df: pd.DataFrame) -> pd.Series:
        is_atm = df.get("tx_type_ATM", pd.Series(0, index=df.index)) == 1
        near_threshold = df["amount"].between(8000, 9799)
        return (is_atm & near_threshold).astype(int)


@dataclass
class HighRiskMccRule(Rule):
    """Transaction at a money-transfer MCC (Western Union, MoneyGram)."""
    name: str = "high_risk_mcc"
    description: str = "Transaction at high-risk MCC (money transfer, gambling)"
    weight: float = 0.6

    def apply(self, df: pd.DataFrame) -> pd.Series:
        return df.get("mcc_high_risk", pd.Series(0, index=df.index)).astype(int)


class RulesEngine:
    """
    Configurable rules engine that applies multiple rules and returns
    individual flags plus a weighted aggregate score.

    In production, rules are loaded from detection_policy.yaml so
    compliance teams can update thresholds without code deploys.
    """

    DEFAULT_RULES: list[Rule] = [
        NightLargeWireRule(),
        RapidSuccessionRule(),
        GeographicImpossibilityRule(),
        StructuringRule(),
        HighRiskMccRule(),
    ]

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules = rules or self.DEFAULT_RULES

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns df with one flag column per rule + 'rules_score' [0,1].
        """
        result = df.copy()
        total_weight = sum(r.weight for r in self.rules)
        rule_flags = []

        for rule in self.rules:
            col = f"rule_{rule.name}"
            result[col] = rule.apply(df)
            rule_flags.append((col, rule.weight))
            log.debug("rule_applied", rule=rule.name, flagged=int(result[col].sum()))

        # Weighted average score
        weighted_sum = sum(result[col] * w for col, w in rule_flags)
        result["rules_score"] = (weighted_sum / total_weight).clip(0, 1)
        return result


# ── 5. Statistical ensemble ───────────────────────────────────────────────────

def statistical_score(df: pd.DataFrame) -> pd.Series:
    """
    Combine all statistical signals into a single normalised score [0, 1].
    This is the 'statistical_score' input to the decision engine.

    Weights: EWMA (0.4) + Z-score (0.3) + rules (0.3)
    EWMA gets highest weight because it adapts to user behaviour over time.
    """
    s_zscore = zscore_score(df)
    s_ewma = ewma_score(df)

    engine = RulesEngine()
    df_rules = engine.apply(df)
    s_rules = df_rules["rules_score"]

    score = 0.3 * s_zscore + 0.4 * s_ewma + 0.3 * s_rules
    return score.clip(0, 1)

