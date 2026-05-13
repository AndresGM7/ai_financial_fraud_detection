"""
llm/llm_client.py
─────────────────
Provider-agnostic LLM client with retry, latency tracking, and cost logging.

Supported backends
──────────────────
  • openai   — GPT-4o (low latency, cheap, great JSON mode)
  • bedrock  — AWS Bedrock Claude (stays in VPC, good for bank compliance)
  • mock     — deterministic stub for tests / CI (no API key needed)

Interview talking points
────────────────────────
  "In production I use an abstraction layer over LLM providers for two
   reasons: (1) it lets us swap between OpenAI and Bedrock without
   changing any detection logic — critical during AWS procurement cycles;
   (2) it's the single place where we enforce retry logic, timeout limits,
   circuit breakers, and cost recording.

  Exponential backoff with jitter is the standard retry strategy for LLM
  rate-limit errors (HTTP 429). Without jitter, all parallel requests
  retry simultaneously and hammer the API again.

  I log every call's token counts to the CostTracker so our ML dashboard
  shows cost-per-alert, cost-per-day, and remaining daily budget in real time."
"""

from __future__ import annotations

import json
import time
from typing import Any

from src.config import settings
from src.utils import cost_tracker, get_logger

log = get_logger(__name__)

# ── Lazy imports (avoid crashing if not installed) ───────────────────────────

def _get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=settings.openai_api_key)


def _get_bedrock_client():
    import boto3
    return boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_default_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )


# ── Mock response (for tests / local dev without API key) ─────────────────────

_MOCK_RESPONSE = {
    "risk": "medium",
    "confidence": 0.65,
    "primary_pattern": None,
    "reason": "Mock response: statistical signals suggest moderate anomaly.",
    "action": "monitor",
    "escalate_to_human": False,
}


# ── Core client ──────────────────────────────────────────────────────────────

class LLMClient:
    """
    Provider-agnostic LLM chat client.

    Usage
    -----
    >>> client = LLMClient()
    >>> response = client.chat(system_prompt, user_message)
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self.provider = provider or settings.llm_provider
        self.model = model or settings.openai_model
        self._openai_client = None
        self._bedrock_client = None

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        max_retries: int = 3,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """
        Send a chat completion request and return parsed JSON response.

        Retries with exponential backoff + jitter on transient errors.
        Records token usage in global CostTracker.
        """
        if cost_tracker.is_over_budget():
            log.warning("llm_budget_exceeded", total_cost=cost_tracker.total_cost_usd)
            return {**_MOCK_RESPONSE, "reason": "Budget limit reached — defaulting to monitor."}

        temperature = temperature if temperature is not None else settings.llm_temperature
        last_exc: Exception | None = None

        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()
                raw, input_tokens, output_tokens = self._dispatch(
                    system_prompt, user_message, temperature
                )
                latency_ms = (time.perf_counter() - t0) * 1000

                cost = cost_tracker.record(self.model, input_tokens, output_tokens)
                log.info(
                    "llm_call",
                    provider=self.provider,
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=round(cost, 6),
                    latency_ms=round(latency_ms, 1),
                    attempt=attempt + 1,
                )
                return _parse_json_response(raw)

            except Exception as exc:
                last_exc = exc
                wait = (2 ** attempt) * 0.5 + (time.time() % 0.3)  # jitter
                log.warning(
                    "llm_retry",
                    attempt=attempt + 1,
                    error=str(exc),
                    wait_s=round(wait, 2),
                )
                time.sleep(wait)

        log.error("llm_failed_all_retries", error=str(last_exc))
        return {**_MOCK_RESPONSE, "reason": f"LLM error after {max_retries} retries."}

    def _dispatch(
        self,
        system: str,
        user: str,
        temperature: float,
    ) -> tuple[str, int, int]:
        """Route to correct provider. Returns (raw_text, input_tokens, output_tokens)."""
        if self.provider == "mock":
            return json.dumps(_MOCK_RESPONSE), 100, 50

        elif self.provider == "openai":
            return self._call_openai(system, user, temperature)

        elif self.provider == "bedrock":
            return self._call_bedrock(system, user, temperature)

        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _call_openai(self, system: str, user: str, temperature: float) -> tuple[str, int, int]:
        if self._openai_client is None:
            self._openai_client = _get_openai_client()

        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=settings.llm_max_tokens_output,
            response_format={"type": "json_object"},  # JSON mode
        )
        content = response.choices[0].message.content
        usage = response.usage
        return content, usage.prompt_tokens, usage.completion_tokens

    def _call_bedrock(self, system: str, user: str, temperature: float) -> tuple[str, int, int]:
        if self._bedrock_client is None:
            self._bedrock_client = _get_bedrock_client()

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": settings.llm_max_tokens_output,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        })
        response = self._bedrock_client.invoke_model(
            modelId=settings.aws_bedrock_model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        content = result["content"][0]["text"]
        usage = result.get("usage", {})
        return content, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


# ── JSON parsing helper ───────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict[str, Any]:
    """
    Extract JSON from LLM response. Handles markdown code fences.
    Falls back to mock on parse failure.
    """
    try:
        # Strip markdown fences if present (common with some models)
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        log.error("json_parse_failed", raw=raw[:200], error=str(e))
        return {**_MOCK_RESPONSE, "reason": "LLM returned invalid JSON — defaulting."}


# Module-level singleton
llm_client = LLMClient()

