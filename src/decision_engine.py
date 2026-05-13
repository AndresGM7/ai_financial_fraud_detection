"""
decision_engine.py
──────────────────
Signal fusion, threshold enforcement, and alert suppression.

Responsibilities
───────────────
  1. Combine statistical / unsupervised / supervised / LLM scores
     into a single weighted risk score.
  2. Apply policy-defined boost rules (from YAML config).
  3. Enforce alert cooldown (prevent alert fatigue for same user).
  4. Emit a final decision: ignore | monitor | alert.

Interview talking points
────────────────────────
  "The decision engine is the 'brain' that converts heterogeneous signals
   into an actionable outcome. The key design decisions are:

  (1) Weighted fusion, not majority vote: ensemble members have different
      precision/recall profiles. LLM gets the highest weight (0.40) because
      it brings contextual reasoning unavailable to tabular models.
      Statistical gets 0.20 — it's a fast, cheap filter, not a final judge.

  (2) Additive boosts for hard signals: geo_impossible is near-deterministic
      (speed-of-light constraint) so I apply a hard 0.25 boost regardless
      of the weighted score. These are the 'slam dunk' rules.

  (3) Alert cooldown: if we've already sent an alert for user U yesterday
      and they trigger again today with moderate score, we shouldn't spam
      the family. Cooldown windows prevent alert fatigue — a real operational
      concern in production fraud systems.

  (4) Policy-as-config: thresholds and weights live in YAML so compliance
      and ops teams can tune without code changes. I track every config
      version in Git for auditability (SOX compliance requirement)."
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from src.config import settings
from src.llm.output_parser import LLMAssessment, RiskLevel
from src.utils import get_logger

log = get_logger(__name__)


# ── Load policy ───────────────────────────────────────────────────────────────

def _load_policy(path: Path = Path("config/detection_policy.yaml")) -> dict:
    try:
        return yaml.safe_load(path.read_text())
    except FileNotFoundError:
        log.warning("policy_file_not_found", path=str(path), using="defaults")
        return {}


# ── Decision output schema ────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    tx_id: str
    user_id: str
    raw_score: float           # weighted composite before boosts
    final_score: float         # after boosts, clipped to [0, 1]
    decision: str              # "ignore" | "monitor" | "alert"
    signal_breakdown: dict[str, float] = field(default_factory=dict)
    applied_boosts: list[str] = field(default_factory=list)
    llm_assessment: dict[str, Any] = field(default_factory=dict)
    suppressed: bool = False   # True if within cooldown window
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "tx_id": self.tx_id,
            "user_id": self.user_id,
            "raw_score": round(self.raw_score, 4),
            "final_score": round(self.final_score, 4),
            "decision": self.decision,
            "signal_breakdown": {k: round(v, 4) for k, v in self.signal_breakdown.items()},
            "applied_boosts": self.applied_boosts,
            "llm_assessment": self.llm_assessment,
            "suppressed": self.suppressed,
            "timestamp": self.timestamp,
        }


# ── Decision engine ───────────────────────────────────────────────────────────

class DecisionEngine:
    """
    Combines all detection signals into a final risk decision.

    Parameters
    ----------
    policy_path : Path to YAML policy file (optional, uses settings defaults)
    """

    def __init__(self, policy_path: Path | None = None) -> None:
        self.policy = _load_policy(policy_path or Path("config/detection_policy.yaml"))
        self._alert_log: dict[str, datetime] = {}  # {user_id: last_alert_time}

        # Load weights from policy or settings
        weights = self.policy.get("signal_weights", {})
        self.w_stat = weights.get("statistical", settings.weight_statistical)
        self.w_unsup = weights.get("unsupervised", settings.weight_unsupervised)
        self.w_sup = weights.get("supervised", settings.weight_supervised)
        self.w_llm = weights.get("llm", settings.weight_llm)

        thresh = self.policy.get("thresholds", {})
        self.alert_threshold = thresh.get("alert", settings.alert_score_threshold)
        self.monitor_threshold = thresh.get("monitor", settings.monitor_score_threshold)
        self.cooldown_hours = thresh.get("cooldown_hours", settings.alert_cooldown_hours)

        self._boosts = self.policy.get("boosts", {})
        self._llm_boosts = self.policy.get("llm_risk_boost", {
            "high": 0.50, "medium": 0.15, "low": 0.00
        })

        log.info(
            "decision_engine_initialised",
            w_stat=self.w_stat, w_unsup=self.w_unsup,
            w_sup=self.w_sup, w_llm=self.w_llm,
            alert_threshold=self.alert_threshold,
        )

    def decide(
        self,
        tx: dict[str, Any],
        statistical_score: float,
        unsupervised_score: float,
        supervised_score: float | None,
        llm_assessment: LLMAssessment | None,
    ) -> DetectionResult:
        """
        Fuse all scores into a final decision.

        Parameters
        ----------
        tx                  : enriched transaction dict (for boost conditions)
        statistical_score   : output from statistical.statistical_score() [0,1]
        unsupervised_score  : output from unsupervised ensemble [0,1]
        supervised_score    : XGBoost/LGBM predict_proba [0,1] or None
        llm_assessment      : parsed LLM output or None
        """
        tx_id = tx.get("tx_id", "unknown")
        user_id = tx.get("user_id", "unknown")

        # ── Step 1: Weighted combination ──────────────────────────────────────
        # When supervised score unavailable (no labels), redistribute its weight
        if supervised_score is None:
            total = self.w_stat + self.w_unsup + self.w_llm
            w_s = self.w_stat / total
            w_u = self.w_unsup / total
            w_l = self.w_llm / total
            raw_score = w_s * statistical_score + w_u * unsupervised_score
            sup_contrib = 0.0
        else:
            raw_score = (
                self.w_stat * statistical_score +
                self.w_unsup * unsupervised_score +
                self.w_sup * supervised_score
            )
            sup_contrib = self.w_sup * supervised_score
            w_l = self.w_llm

        # LLM contribution
        llm_score = 0.0
        llm_dict: dict = {}
        if llm_assessment:
            llm_score = llm_assessment.risk_score()
            raw_score += w_l * llm_score
            llm_dict = llm_assessment.to_dict()

        raw_score = float(min(raw_score, 1.0))

        signal_breakdown = {
            "statistical": statistical_score * self.w_stat,
            "unsupervised": unsupervised_score * self.w_unsup,
            "supervised": sup_contrib,
            "llm": w_l * llm_score,
        }

        # ── Step 2: Apply boosts ───────────────────────────────────────────────
        final_score = raw_score
        applied_boosts: list[str] = []

        boost_conditions = {
            "new_counterparty": tx.get("is_new_counterparty", 0) == 1,
            "night_large_wire": (
                tx.get("is_night", 0) == 1 and
                tx.get("tx_type", "") == "WIRE" and
                tx.get("amount", 0) > tx.get("amount_mean_7tx", 0) * 3
            ),
            "rapid_succession": tx.get("tx_count_1h", 0) > 5,
            "geographic_velocity": tx.get("geo_impossible", 0) == 1,
        }

        for boost_name, condition in boost_conditions.items():
            if condition:
                boost_val = self._boosts.get(boost_name, 0.0)
                final_score += boost_val
                applied_boosts.append(f"{boost_name}+{boost_val:.2f}")

        # LLM risk-level boost (on top of weighted score)
        if llm_assessment:
            risk_key = llm_assessment.risk.value
            llm_boost = self._llm_boosts.get(risk_key, 0.0)
            if llm_boost > 0:
                final_score += llm_boost
                applied_boosts.append(f"llm_risk_{risk_key}+{llm_boost:.2f}")

        final_score = float(min(final_score, 1.0))

        # ── Step 3: Threshold decision ────────────────────────────────────────
        if final_score >= self.alert_threshold:
            decision = "alert"
        elif final_score >= self.monitor_threshold:
            decision = "monitor"
        else:
            decision = "ignore"

        # ── Step 4: Cooldown suppression ──────────────────────────────────────
        suppressed = False
        if decision == "alert":
            last_alert = self._alert_log.get(user_id)
            if last_alert:
                elapsed = datetime.now(timezone.utc) - last_alert
                if elapsed < timedelta(hours=self.cooldown_hours):
                    log.info(
                        "alert_suppressed_cooldown",
                        user_id=user_id,
                        elapsed_h=elapsed.total_seconds() / 3600,
                    )
                    decision = "monitor"
                    suppressed = True
            if not suppressed:
                self._alert_log[user_id] = datetime.now(timezone.utc)

        log.info(
            "decision_made",
            tx_id=tx_id,
            raw_score=round(raw_score, 4),
            final_score=round(final_score, 4),
            decision=decision,
            boosts=applied_boosts,
            suppressed=suppressed,
        )

        return DetectionResult(
            tx_id=tx_id,
            user_id=user_id,
            raw_score=raw_score,
            final_score=final_score,
            decision=decision,
            signal_breakdown=signal_breakdown,
            applied_boosts=applied_boosts,
            llm_assessment=llm_dict,
            suppressed=suppressed,
        )

    def reset_cooldowns(self) -> None:
        """Clear cooldown log (used in testing and daily resets)."""
        self._alert_log.clear()

