"""
Prompt Builder: construct structured prompts for the LLM risk assessment.
"""


def build_prompt(tx: dict, context: dict) -> str:
    """
    Build a structured prompt for the LLM to assess transaction risk.

    Args:
        tx: Transaction dictionary with fields: amount, merchant, hour, etc.
        context: User behavioral context from the RAG retriever.

    Returns:
        A formatted prompt string.
    """
    common_merchants = context.get("common_merchants", [])
    merchants_str = ", ".join(common_merchants) if common_merchants else "N/A"

    return f"""You are a financial safety assistant. Analyze the following transaction and assess its fraud risk.

User profile (last 50 transactions):
- Average transaction amount: ${context.get('avg_amount', 0):.2f}
- Std deviation of amount: ${context.get('std_amount', 0):.2f}
- Most common merchants: {merchants_str}
- Night-time activity ratio: {context.get('night_activity', 0):.1%}
- Weekend activity ratio: {context.get('weekend_activity', 0):.1%}
- Total transactions analyzed: {context.get('tx_count', 0)}

Current transaction:
- Amount: ${tx.get('amount', 0):.2f}
- Merchant: {tx.get('merchant', 'unknown')}
- Hour of day: {tx.get('hour', 0):02d}:00
- Day of week: {tx.get('day_of_week', 'unknown')}
- Is night transaction: {bool(tx.get('is_night', 0))}
- Is weekend: {bool(tx.get('is_weekend', 0))}

Based on the user's behavioral profile and this transaction, assess the fraud risk.

Respond with a JSON object only (no markdown, no extra text):
{{
  "risk": "low|medium|high",
  "reason": "brief explanation of risk assessment",
  "action": "ignore|monitor|alert",
  "confidence": 0.0
}}"""


def build_system_prompt() -> str:
    """Return the system-level prompt for the LLM."""
    return (
        "You are an expert financial fraud detection assistant. "
        "You analyze transaction patterns and user behavior to identify potential fraud. "
        "Always respond with valid JSON only. "
        "Be concise and precise in your risk assessments."
    )
