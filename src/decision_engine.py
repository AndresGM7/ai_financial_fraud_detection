"""
Decision Engine: fuse signals from all detection layers into a final decision.
"""
import logging
from typing import Optional

from src.config import (
    WEIGHT_STATISTICAL,
    WEIGHT_UNSUPERVISED,
    WEIGHT_SUPERVISED,
    WEIGHT_LLM_HIGH,
    ALERT_THRESHOLD,
    MONITOR_THRESHOLD,
)

logger = logging.getLogger(__name__)

_RISK_SCORE_MAP = {"low": 0.0, "medium": 0.3, "high": 1.0}


def combine_signals(
    stat: float,
    unsup: float,
    sup: Optional[float],
    llm: dict,
) -> dict:
    """
    Combine signals from all detection layers into a final risk decision.

    Args:
        stat: Statistical layer score (0 or 1).
        unsup: Unsupervised layer score (0 or 1).
        sup: Supervised layer score (0-1 probability), or None if no labels.
        llm: Parsed LLM output dict with 'risk', 'action', 'reason', 'confidence'.

    Returns:
        dict with keys: 'score' (float), 'decision' (str), 'signals' (dict).
    """
    score = stat * WEIGHT_STATISTICAL + unsup * WEIGHT_UNSUPERVISED

    # Add supervised contribution if available
    if sup is not None:
        score += sup * WEIGHT_SUPERVISED

    # Add LLM contribution
    llm_risk = llm.get("risk", "low")
    llm_score = _RISK_SCORE_MAP.get(llm_risk, 0.0)
    if llm_risk == "high":
        score += WEIGHT_LLM_HIGH
    else:
        score += llm_score * 0.2

    score = min(score, 1.0)

    if score >= ALERT_THRESHOLD:
        decision = "alert"
    elif score >= MONITOR_THRESHOLD:
        decision = "monitor"
    else:
        decision = "ignore"

    logger.debug(
        "Decision: %s (score=%.3f, stat=%s, unsup=%s, sup=%s, llm=%s)",
        decision,
        score,
        stat,
        unsup,
        sup,
        llm_risk,
    )

    return {
        "score": round(score, 4),
        "decision": decision,
        "signals": {
            "statistical": int(stat),
            "unsupervised": int(unsup),
            "supervised": float(sup) if sup is not None else None,
            "llm_risk": llm_risk,
            "llm_reason": llm.get("reason", ""),
            "llm_action": llm.get("action", ""),
        },
    }
