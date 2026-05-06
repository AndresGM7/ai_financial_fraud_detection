"""
Output Parser: parse and validate LLM JSON responses.
"""
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

VALID_RISKS = {"low", "medium", "high"}
VALID_ACTIONS = {"ignore", "monitor", "alert"}

_DEFAULT_RESPONSE = {
    "risk": "medium",
    "reason": "Unable to parse LLM response",
    "action": "monitor",
    "confidence": 0.5,
}


def parse_llm_output(raw: str) -> dict[str, Any]:
    """
    Parse and validate the LLM JSON output.

    Handles cases where the model wraps JSON in markdown code blocks.
    Falls back to a safe default if parsing fails.

    Args:
        raw: Raw string from the LLM.

    Returns:
        Validated dictionary with risk, reason, action, and confidence.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM output as JSON: %r", raw)
        return _DEFAULT_RESPONSE.copy()

    risk = str(data.get("risk", "medium")).lower()
    action = str(data.get("action", "monitor")).lower()

    if risk not in VALID_RISKS:
        logger.warning("Invalid risk value %r, defaulting to 'medium'", risk)
        risk = "medium"
    if action not in VALID_ACTIONS:
        logger.warning("Invalid action value %r, defaulting to 'monitor'", action)
        action = "monitor"

    try:
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    return {
        "risk": risk,
        "reason": str(data.get("reason", "")),
        "action": action,
        "confidence": confidence,
    }
