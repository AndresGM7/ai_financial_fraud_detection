"""
Tests for LLM modules: output parser, prompt builder, and RAG retriever.
"""
import datetime
import json

import pandas as pd
import pytest

from src.llm.output_parser import parse_llm_output
from src.llm.prompt_builder import build_prompt, build_system_prompt
from src.llm.rag_retriever import retrieve_context
from src.enrichment import enrich_transactions


def make_enriched_df(n: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "transaction_id": f"tx_{i}",
            "user_id": "user_1" if i < 15 else "user_2",
            "timestamp": (
                datetime.datetime(2024, 1, 1) + datetime.timedelta(hours=i * 3)
            ).isoformat(),
            "amount": float(100 + i * 10),
            "merchant": ["Amazon", "Walmart", "Netflix"][i % 3],
        })
    return enrich_transactions(pd.DataFrame(rows))


class TestOutputParser:
    def test_valid_json_low_risk(self):
        raw = json.dumps({"risk": "low", "reason": "Normal tx", "action": "ignore", "confidence": 0.1})
        result = parse_llm_output(raw)
        assert result["risk"] == "low"
        assert result["action"] == "ignore"

    def test_valid_json_high_risk(self):
        raw = json.dumps({"risk": "high", "reason": "Suspicious", "action": "alert", "confidence": 0.9})
        result = parse_llm_output(raw)
        assert result["risk"] == "high"
        assert result["action"] == "alert"

    def test_invalid_json_returns_default(self):
        result = parse_llm_output("not json at all")
        assert result["risk"] == "medium"
        assert result["action"] == "monitor"

    def test_strips_markdown_code_fences(self):
        raw = "```json\n{\"risk\": \"low\", \"reason\": \"OK\", \"action\": \"ignore\", \"confidence\": 0.2}\n```"
        result = parse_llm_output(raw)
        assert result["risk"] == "low"

    def test_invalid_risk_value_defaults_to_medium(self):
        raw = json.dumps({"risk": "extreme", "reason": "X", "action": "ignore", "confidence": 0.5})
        result = parse_llm_output(raw)
        assert result["risk"] == "medium"

    def test_confidence_clamped(self):
        raw = json.dumps({"risk": "low", "reason": "OK", "action": "ignore", "confidence": 99.0})
        result = parse_llm_output(raw)
        assert result["confidence"] <= 1.0


class TestPromptBuilder:
    def test_build_prompt_contains_amount(self):
        tx = {"amount": 500.0, "merchant": "Amazon", "hour": 14, "day_of_week": 2, "is_night": 0, "is_weekend": 0}
        context = {"avg_amount": 100.0, "std_amount": 20.0, "common_merchants": ["amazon"], "night_activity": 0.1, "weekend_activity": 0.2, "tx_count": 50}
        prompt = build_prompt(tx, context)
        assert "500.00" in prompt
        assert "Amazon" in prompt

    def test_build_system_prompt_returns_string(self):
        sp = build_system_prompt()
        assert isinstance(sp, str)
        assert len(sp) > 0


class TestRagRetriever:
    def test_returns_dict_with_expected_keys(self):
        df = make_enriched_df()
        ctx = retrieve_context("user_1", df)
        for key in ["avg_amount", "std_amount", "common_merchants", "night_activity", "tx_count"]:
            assert key in ctx, f"Missing key: {key}"

    def test_unknown_user_returns_empty_context(self):
        df = make_enriched_df()
        ctx = retrieve_context("nonexistent_user", df)
        assert ctx["tx_count"] == 0
        assert ctx["avg_amount"] == 0.0

    def test_common_merchants_is_list(self):
        df = make_enriched_df()
        ctx = retrieve_context("user_1", df)
        assert isinstance(ctx["common_merchants"], list)
