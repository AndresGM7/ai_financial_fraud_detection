"""
tests/test_decision_engine.py
──────────────────────────────
Unit tests for decision_engine.py.
"""

from __future__ import annotations

import pytest

from src.decision_engine import DecisionEngine
from src.llm.output_parser import ActionLevel, LLMAssessment, RiskLevel


@pytest.fixture
def engine() -> DecisionEngine:
    eng = DecisionEngine()
    eng.reset_cooldowns()
    return eng


@pytest.fixture
def normal_tx() -> dict:
    return {
        "tx_id": "TX00000001",
        "user_id": "U00001",
        "amount": 100.0,
        "amount_mean_7tx": 100.0,
        "tx_type": "POS",
        "is_night": 0,
        "is_new_counterparty": 0,
        "geo_impossible": 0,
        "tx_count_1h": 1.0,
    }


@pytest.fixture
def high_risk_tx() -> dict:
    return {
        "tx_id": "TX00000002",
        "user_id": "U00002",
        "amount": 9500.0,
        "amount_mean_7tx": 100.0,
        "tx_type": "WIRE",
        "is_night": 1,
        "is_new_counterparty": 1,
        "geo_impossible": 1,
        "tx_count_1h": 8.0,
    }


@pytest.fixture
def high_llm() -> LLMAssessment:
    return LLMAssessment(
        risk=RiskLevel.HIGH,
        confidence=0.9,
        primary_pattern="romance_scam",
        reason="Large first-time wire to unknown overseas entity at night.",
        action=ActionLevel.ALERT,
        escalate_to_human=True,
    )


@pytest.fixture
def low_llm() -> LLMAssessment:
    return LLMAssessment(
        risk=RiskLevel.LOW,
        confidence=0.85,
        primary_pattern=None,
        reason="Transaction consistent with user's normal spending behaviour.",
        action=ActionLevel.IGNORE,
        escalate_to_human=False,
    )


class TestDecisionEngine:
    def test_low_scores_ignored(self, engine, normal_tx, low_llm):
        result = engine.decide(
            tx=normal_tx,
            statistical_score=0.05,
            unsupervised_score=0.05,
            supervised_score=0.02,
            llm_assessment=low_llm,
        )
        assert result.decision == "ignore"

    def test_high_scores_alerted(self, engine, high_risk_tx, high_llm):
        result = engine.decide(
            tx=high_risk_tx,
            statistical_score=0.9,
            unsupervised_score=0.85,
            supervised_score=0.88,
            llm_assessment=high_llm,
        )
        assert result.decision == "alert"
        assert result.final_score >= engine.alert_threshold

    def test_geo_impossible_boost_applied(self, engine, normal_tx, low_llm):
        tx = {**normal_tx, "geo_impossible": 1}
        result = engine.decide(
            tx=tx,
            statistical_score=0.3,
            unsupervised_score=0.3,
            supervised_score=0.3,
            llm_assessment=low_llm,
        )
        assert "geographic_velocity" in " ".join(result.applied_boosts)

    def test_score_never_exceeds_one(self, engine, high_risk_tx, high_llm):
        result = engine.decide(
            tx=high_risk_tx,
            statistical_score=1.0,
            unsupervised_score=1.0,
            supervised_score=1.0,
            llm_assessment=high_llm,
        )
        assert result.final_score <= 1.0

    def test_cooldown_suppresses_second_alert(self, engine, high_risk_tx, high_llm):
        # First alert
        r1 = engine.decide(
            tx=high_risk_tx,
            statistical_score=0.9,
            unsupervised_score=0.85,
            supervised_score=0.88,
            llm_assessment=high_llm,
        )
        assert r1.decision == "alert"
        assert not r1.suppressed

        # Second alert same user — should be suppressed
        r2 = engine.decide(
            tx={**high_risk_tx, "tx_id": "TX00000003"},
            statistical_score=0.9,
            unsupervised_score=0.85,
            supervised_score=0.88,
            llm_assessment=high_llm,
        )
        assert r2.suppressed
        assert r2.decision == "monitor"

    def test_no_supervised_score_redistributes_weight(self, engine, normal_tx, low_llm):
        """When supervised score is None, weights should still produce valid score."""
        result = engine.decide(
            tx=normal_tx,
            statistical_score=0.5,
            unsupervised_score=0.5,
            supervised_score=None,
            llm_assessment=low_llm,
        )
        assert 0.0 <= result.final_score <= 1.0

    def test_signal_breakdown_sums_to_raw_score(self, engine, normal_tx, low_llm):
        result = engine.decide(
            tx=normal_tx,
            statistical_score=0.3,
            unsupervised_score=0.2,
            supervised_score=0.1,
            llm_assessment=low_llm,
        )
        total = sum(result.signal_breakdown.values())
        assert abs(total - result.raw_score) < 0.02  # small float rounding ok

