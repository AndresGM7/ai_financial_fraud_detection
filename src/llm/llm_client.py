"""
LLM Client: wrapper around OpenAI API for fraud risk reasoning.
"""
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OPENAI_AVAILABLE = False


class LLMClient:
    """Thin wrapper around the OpenAI chat completion API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        if not _OPENAI_AVAILABLE:
            raise ImportError("openai package is required. Run: pip install openai")

        resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key must be set via OPENAI_API_KEY env var or api_key parameter"
            )
        self._client = OpenAI(api_key=resolved_key)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a chat completion request and return the response text.

        Args:
            system_prompt: The system-level instruction.
            user_prompt: The user-level prompt with transaction details.

        Returns:
            Raw response string from the LLM.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content or ""
        logger.debug("LLM response: %s", content)
        return content


class MockLLMClient:
    """Mock LLM client for testing without an API key."""

    def __init__(self, default_risk: str = "low"):
        self.default_risk = default_risk

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return json.dumps({
            "risk": self.default_risk,
            "reason": "Mock assessment for testing",
            "action": "ignore" if self.default_risk == "low" else "monitor",
            "confidence": 0.5,
        })
