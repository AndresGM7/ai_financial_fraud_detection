"""
llm/output_parser.py
─────────────────────
Pydantic-validated structured output from LLM responses.

Interview talking points
────────────────────────
  "Raw LLM output is untyped text — you can't rely on field names, value
   ranges, or enum membership without validation. I use Pydantic as a
   contract layer: the LLM must return exactly the schema I define, or
   I catch the violation and handle it gracefully (fallback, alert).

  This pattern ('structured outputs') is now natively supported in
  OpenAI's API (response_format=json_schema) and Anthropic's tool_use
  feature. Using Pydantic for validation is idiomatic regardless of
  which mechanism delivers the JSON."
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.utils import get_logger

log = get_logger(__name__)


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionLevel(str, Enum):
    IGNORE = "ignore"
    MONITOR = "monitor"
    ALERT = "alert"


class LLMAssessment(BaseModel):
    """
    Validated, typed LLM output.

    Fields
    ──────
    risk            : Overall risk level
    confidence      : Model confidence in the assessment [0.0, 1.0]
    primary_pattern : ID of the most likely fraud typology (or None)
    reason          : One-sentence human-readable explanation
    action          : Recommended action
    escalate_to_human: Whether a human analyst should review
    """
    risk: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)
    primary_pattern: str | None = None
    reason: str
    action: ActionLevel
    escalate_to_human: bool = False

    @field_validator("reason")
    @classmethod
    def reason_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reason must not be empty")
        return v.strip()

    def risk_score(self) -> float:
        """Convert categorical risk to numeric score for decision engine."""
        return {
            RiskLevel.LOW: 0.1,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.9,
        }[self.risk] * self.confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk": self.risk.value,
            "confidence": self.confidence,
            "primary_pattern": self.primary_pattern,
            "reason": self.reason,
            "action": self.action.value,
            "escalate_to_human": self.escalate_to_human,
            "risk_score": self.risk_score(),
        }


# ── Default fallback ──────────────────────────────────────────────────────────

DEFAULT_ASSESSMENT = LLMAssessment(
    risk=RiskLevel.MEDIUM,
    confidence=0.5,
    primary_pattern=None,
    reason="LLM assessment unavailable — defaulting to medium risk for safety.",
    action=ActionLevel.MONITOR,
    escalate_to_human=False,
)


def parse_llm_output(raw: dict[str, Any]) -> LLMAssessment:
    """
    Parse and validate raw LLM JSON dict into a typed LLMAssessment.
    Falls back to DEFAULT_ASSESSMENT on validation errors.
    """
    try:
        return LLMAssessment.model_validate(raw)
    except Exception as exc:
        log.warning("llm_output_parse_error", error=str(exc), raw=raw)
        return DEFAULT_ASSESSMENT

