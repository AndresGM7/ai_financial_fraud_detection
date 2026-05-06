"""
Tests for the decision engine.
"""
import pytest
from src.decision_engine import combine_signals


class TestCombineSignals:
    def _llm(self, risk: str = "low") -> dict:
        actions = {"low": "ignore", "medium": "monitor", "high": "alert"}
        return {
            "risk": risk,
            "reason": "test",
            "action": actions[risk],
            "confidence": 0.5,
        }

    def test_all_zero_signals_is_ignore(self):
        result = combine_signals(0, 0, None, self._llm("low"))
        assert result["decision"] == "ignore"
        assert result["score"] < 0.4

    def test_all_max_signals_is_alert(self):
        result = combine_signals(1, 1, 1.0, self._llm("high"))
        assert result["decision"] == "alert"
        assert result["score"] >= 0.7

    def test_llm_high_risk_boosts_score(self):
        result_low = combine_signals(0, 0, None, self._llm("low"))
        result_high = combine_signals(0, 0, None, self._llm("high"))
        assert result_high["score"] > result_low["score"]

    def test_no_supervised_signal(self):
        result = combine_signals(1, 1, None, self._llm("low"))
        assert result["signals"]["supervised"] is None

    def test_decision_keys_present(self):
        result = combine_signals(0, 0, None, self._llm("low"))
        assert "score" in result
        assert "decision" in result
        assert "signals" in result

    def test_score_clamped_to_one(self):
        result = combine_signals(1, 1, 1.0, self._llm("high"))
        assert result["score"] <= 1.0

    def test_monitor_threshold(self):
        # stat=1 + unsup=1 with medium llm risk should give monitor or alert
        result = combine_signals(1, 1, None, self._llm("medium"))
        assert result["decision"] in ("monitor", "alert")
