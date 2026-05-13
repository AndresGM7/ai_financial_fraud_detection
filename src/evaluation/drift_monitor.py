"""
evaluation/drift_monitor.py
────────────────────────────
Production feature drift detection.

Drift signals
─────────────
  • Population Stability Index (PSI) — detects distribution shift vs. baseline
  • KL-divergence                    — information-theoretic distance measure
  • Jensen-Shannon divergence        — symmetric, bounded version of KL
  • Chi-squared test                 — for categorical feature drift

Interview talking points
────────────────────────
  "Models decay in production — a phenomenon called 'concept drift'. In
   fraud detection, drift has two causes:
   (1) Data drift: transaction distributions change (new merchants, COVID).
   (2) Concept drift: the relationship between features and fraud changes
       (fraudsters adapt to the detection system).

  PSI is the industry standard for drift detection at banks and fintechs.
  PSI < 0.1 → no significant shift, no action needed.
  PSI 0.1–0.2 → moderate shift, monitor closely.
  PSI > 0.2 → significant shift, retrain model.

  I monitor PSI weekly for top features (Z-score, amount, tx_count_24h)
  and daily for the model's score distribution. A sudden jump in the
  average fraud score on a Monday morning likely means a data pipeline
  problem, not an actual fraud surge."
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import get_logger

log = get_logger(__name__)


# ── PSI ───────────────────────────────────────────────────────────────────────

def psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Population Stability Index.

    PSI = Σ (actual% - expected%) * ln(actual% / expected%)

    Interpretation:
      PSI < 0.10  : No significant change — model OK
      0.10–0.20  : Moderate change — monitor
      > 0.20     : Significant change — retrain

    Interview talking point:
      "PSI was developed by the insurance and credit risk industry in the
       1970s. It's still the go-to metric because it's fast, interpretable,
       and doesn't require the full test-set rerun. I compute it on the
       TOP 10 features by SHAP importance — monitoring all 30 features is
       noisy and expensive."
    """
    # Remove NaN and clip to valid range
    base = np.array(baseline, dtype=float)
    curr = np.array(current, dtype=float)
    base = base[~np.isnan(base)]
    curr = curr[~np.isnan(curr)]

    # Compute bin edges from baseline
    bin_edges = np.percentile(base, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] -= epsilon
    bin_edges[-1] += epsilon

    # Proportions
    base_counts = np.histogram(base, bins=bin_edges)[0]
    curr_counts = np.histogram(curr, bins=bin_edges)[0]

    base_pct = base_counts / (len(base) + epsilon)
    curr_pct = curr_counts / (len(curr) + epsilon)

    # Add epsilon to avoid log(0)
    base_pct = np.where(base_pct == 0, epsilon, base_pct)
    curr_pct = np.where(curr_pct == 0, epsilon, curr_pct)

    psi_value = float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))
    return round(psi_value, 6)


def psi_flag(psi_value: float) -> str:
    if psi_value < 0.10:
        return "stable"
    elif psi_value < 0.20:
        return "moderate_shift"
    else:
        return "significant_shift"


# ── KL Divergence ─────────────────────────────────────────────────────────────

def kl_divergence(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 20,
    epsilon: float = 1e-6,
) -> float:
    """
    KL(current || baseline) — asymmetric measure of distribution shift.
    KL = 0 means identical distributions. Higher = more shift.
    """
    base = np.array(baseline, dtype=float)
    curr = np.array(current, dtype=float)
    bin_edges = np.percentile(base[~np.isnan(base)], np.linspace(0, 100, n_bins + 1))
    bin_edges[0] -= epsilon
    bin_edges[-1] += epsilon

    p = np.histogram(base, bins=bin_edges)[0].astype(float) + epsilon
    q = np.histogram(curr, bins=bin_edges)[0].astype(float) + epsilon
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(q * np.log(q / p)))


# ── Jensen-Shannon divergence ─────────────────────────────────────────────────

def js_divergence(baseline: np.ndarray, current: np.ndarray, **kwargs) -> float:
    """Symmetric, bounded [0, ln2] version of KL-divergence."""
    return float(stats.entropy(
        _to_probs(baseline, current, **kwargs),
        base=None
    ))


def _to_probs(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 20,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    bin_edges = np.percentile(
        baseline[~np.isnan(baseline)], np.linspace(0, 100, n_bins + 1)
    )
    bin_edges[0] -= epsilon
    bin_edges[-1] += epsilon
    p = np.histogram(baseline, bins=bin_edges)[0].astype(float) + epsilon
    q = np.histogram(current, bins=bin_edges)[0].astype(float) + epsilon
    return p / p.sum(), q / q.sum()


# ── Chi-squared for categoricals ──────────────────────────────────────────────

def chi2_drift(
    baseline: pd.Series,
    current: pd.Series,
) -> dict[str, float]:
    """
    Chi-squared test for drift in categorical features.
    Returns p-value (< 0.05 = significant drift).
    """
    combined_cats = set(baseline.unique()) | set(current.unique())
    base_counts = baseline.value_counts().reindex(combined_cats, fill_value=0)
    curr_counts = current.value_counts().reindex(combined_cats, fill_value=0)

    chi2, p_value = stats.chisquare(curr_counts.values + 1, base_counts.values + 1)
    return {
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(float(p_value), 6),
        "significant_drift": bool(p_value < 0.05),
    }


# ── Full drift report ─────────────────────────────────────────────────────────

def drift_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numerical_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    """
    Generate a drift report comparing baseline to current production data.

    Returns a DataFrame with PSI, KL-divergence, and drift flag per feature.
    """
    rows = []

    for col in numerical_cols:
        if col not in baseline_df.columns or col not in current_df.columns:
            continue
        base_vals = baseline_df[col].dropna().values
        curr_vals = current_df[col].dropna().values
        psi_val = psi(base_vals, curr_vals)
        kl_val = kl_divergence(base_vals, curr_vals)
        rows.append({
            "feature": col,
            "type": "numerical",
            "psi": psi_val,
            "psi_flag": psi_flag(psi_val),
            "kl_divergence": round(kl_val, 6),
            "significant_drift": psi_val >= 0.20,
        })

    for col in categorical_cols:
        if col not in baseline_df.columns or col not in current_df.columns:
            continue
        chi2_result = chi2_drift(baseline_df[col], current_df[col])
        rows.append({
            "feature": col,
            "type": "categorical",
            "psi": None,
            "psi_flag": None,
            "kl_divergence": None,
            "significant_drift": chi2_result["significant_drift"],
            **chi2_result,
        })

    report_df = pd.DataFrame(rows).sort_values("significant_drift", ascending=False)
    n_drifted = report_df["significant_drift"].sum()
    log.info("drift_report_complete", features=len(rows), drifted=int(n_drifted))
    return report_df

