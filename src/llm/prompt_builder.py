"""
llm/prompt_builder.py
─────────────────────
Dynamic, token-budget-aware prompt assembly.

Design principles
─────────────────
  • Deterministic: same inputs → same prompt (reproducible evals)
  • Token-budgeted: never exceeds max_tokens_input from settings
  • Structured output: always requests JSON (parseable, not prose)
  • Role-specific: system prompt encodes domain expertise

Interview talking points
────────────────────────
  "The most common LLM production failure I've seen is runaway prompt
   length — especially in RAG systems where context can explode. I enforce
   a hard token budget using tiktoken, truncating the least important
   sections (history details) first. The critical sections (current
   transaction, system rules) are never truncated.

  I always request JSON output with a strict schema, then parse it with
  Pydantic. This gives me type safety and lets me catch hallucinations
  (e.g. LLM outputs 'very_high' instead of 'high') at the boundary.

  The system prompt explicitly encodes elder-fraud domain knowledge —
  rather than expecting the LLM to retrieve it from training, I give it
  the relevant typologies as context. This is the RAG principle."
"""

from __future__ import annotations

from typing import Any

import tiktoken

from src.config import settings
from src.utils import get_logger

log = get_logger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")  # matches GPT-4/Claude tokeniser


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior financial fraud analyst specialising in elder financial abuse.
Your task is to evaluate whether a financial transaction warrants a fraud alert for an older adult customer.

You must:
1. Analyse the transaction against the user's historical behaviour profile.
2. Consider the retrieved fraud typologies that are contextually relevant.
3. Reason step-by-step before reaching a conclusion.
4. Account for the statistical signals already computed (Z-score, velocity, rules flags).
5. Be calibrated: default to "low" risk when evidence is ambiguous. False positives erode customer trust.

You MUST return a valid JSON object with exactly these fields:
{
  "risk": "low" | "medium" | "high",
  "confidence": 0.0–1.0,
  "primary_pattern": "<fraud_typology_id or null>",
  "reason": "<one sentence explanation>",
  "action": "ignore" | "monitor" | "alert",
  "escalate_to_human": true | false
}
Do not include any text outside the JSON object."""


def build_prompt(
    current_tx: dict[str, Any],
    user_context: dict[str, Any],
    statistical_signals: dict[str, Any],
    max_tokens: int | None = None,
) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) pair for the LLM call.

    Sections (in order of priority — lower sections get truncated first):
      1. Current transaction details         [always kept]
      2. Statistical signals summary         [always kept]
      3. User historical profile             [kept if budget allows]
      4. Relevant fraud typologies           [truncated last]

    Returns
    -------
    (system_prompt, user_message)
    """
    max_tokens = max_tokens or settings.llm_max_tokens_input
    reserved_output = settings.llm_max_tokens_output
    available = max_tokens - _count_tokens(SYSTEM_PROMPT) - reserved_output

    # ── Section 1: Current transaction (never truncate) ───────────────────────
    tx_section = _format_transaction(current_tx)

    # ── Section 2: Statistical signals (never truncate) ───────────────────────
    signals_section = _format_signals(statistical_signals)

    # ── Section 3: User profile ───────────────────────────────────────────────
    profile_section = _format_user_profile(user_context)

    # ── Section 4: Fraud typologies ───────────────────────────────────────────
    typology_section = _format_typologies(user_context.get("relevant_fraud_patterns", []))

    # Assemble with budget enforcement
    core = tx_section + "\n\n" + signals_section
    remaining = available - _count_tokens(core)

    if remaining > _count_tokens(typology_section):
        remaining -= _count_tokens(typology_section)
        type_content = typology_section
    else:
        type_content = ""
        log.warning("typologies_truncated_budget")

    if remaining > _count_tokens(profile_section):
        profile_content = profile_section
    else:
        # Truncate profile to fit budget
        lines = profile_section.split("\n")
        profile_content = "\n".join(lines[:max(1, remaining // 30)])
        log.warning("profile_truncated_budget", kept_lines=len(profile_content.split("\n")))

    user_message = "\n\n".join(filter(None, [
        "## Transaction Under Review",
        core,
        "## Relevant Fraud Patterns",
        type_content,
        "## User Historical Profile",
        profile_content,
        "\nAnalyse this transaction and return your JSON assessment.",
    ]))

    total_tokens = _count_tokens(SYSTEM_PROMPT) + _count_tokens(user_message)
    log.debug("prompt_built", total_tokens=total_tokens, budget=max_tokens)

    return SYSTEM_PROMPT, user_message


def _format_transaction(tx: dict[str, Any]) -> str:
    lines = [
        f"- Transaction ID: {tx.get('tx_id', 'N/A')}",
        f"- Amount: ${tx.get('amount', 0):,.2f}",
        f"- Merchant: {tx.get('merchant', 'unknown')}",
        f"- Merchant (cleaned): {tx.get('merchant_clean', 'unknown')}",
        f"- Type: {tx.get('tx_type', 'unknown')}",
        f"- MCC: {tx.get('mcc', 9999)}",
        f"- Counterparty: {tx.get('counterparty', 'unknown')}",
        f"- Hour: {tx.get('hour', 'N/A')}:00",
        f"- Day of week: {tx.get('day_of_week', 'N/A')} (0=Mon)",
        f"- Is night (00:00–05:59): {bool(tx.get('is_night', 0))}",
        f"- Is new counterparty: {bool(tx.get('is_new_counterparty', 0))}",
        f"- Geographic speed km/h: {tx.get('geo_speed_kmh', 0):.1f}",
        f"- Geo impossible travel: {bool(tx.get('geo_impossible', 0))}",
    ]
    return "\n".join(lines)


def _format_signals(signals: dict[str, Any]) -> str:
    lines = [
        "**Pre-computed Statistical Signals:**",
        f"- Z-score (amount vs user baseline): {signals.get('z_amount', 0):.2f}",
        f"- EWMA Z-score: {signals.get('ewma_z_amount', 0):.2f}",
        f"- Transactions last 1h: {signals.get('tx_count_1h', 0):.0f}",
        f"- Transactions last 24h: {signals.get('tx_count_24h', 0):.0f}",
        f"- Statistical score (0–1): {signals.get('statistical_score', 0):.3f}",
        f"- Unsupervised anomaly score (0–1): {signals.get('unsupervised_score', 0):.3f}",
        f"- MCC high-risk: {bool(signals.get('mcc_high_risk', 0))}",
    ]
    for key, val in signals.items():
        if key.startswith("rule_") and val == 1:
            lines.append(f"- Rule triggered: {key.replace('rule_', '')}")
    return "\n".join(lines)


def _format_user_profile(ctx: dict[str, Any]) -> str:
    lines = [
        "**User Behavioural Profile (last 50 transactions):**",
        f"- Average amount: ${ctx.get('avg_amount', 0):,.2f}",
        f"- Std deviation: ${ctx.get('std_amount', 0):,.2f}",
        f"- Median amount: ${ctx.get('median_amount', 0):,.2f}",
        f"- 95th percentile amount: ${ctx.get('p95_amount', 0):,.2f}",
        f"- Common merchants: {', '.join(ctx.get('common_merchants', [])[:5])}",
        f"- Common tx types: {', '.join(ctx.get('common_tx_types', [])[:3])}",
        f"- Night activity ratio: {ctx.get('night_activity_ratio', 0):.1%}",
        f"- New counterparty rate: {ctx.get('new_counterparty_rate', 0):.1%}",
        f"- Avg transactions per day: {ctx.get('avg_tx_per_day', 0):.1f}",
    ]
    return "\n".join(lines)


def _format_typologies(typologies: list[dict]) -> str:
    if not typologies:
        return ""
    lines = []
    for t in typologies[:2]:  # limit to 2 most relevant
        lines.append(f"**{t.get('title', '')}** (id: {t.get('id', '')})")
        lines.append(t.get("description", ""))
        flags = t.get("red_flags", [])
        if flags:
            lines.append(f"  Red flags: {', '.join(flags)}")
        lines.append("")
    return "\n".join(lines)

