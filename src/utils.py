"""
utils.py
────────
Shared helpers: structured logging, timing, serialisation, cost tracker.

Interview talking point:
  "I use structlog so every log line is machine-parseable JSON — critical
   for CloudWatch / Datadog dashboards and incident post-mortems."
"""

from __future__ import annotations

import functools
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Generator

import structlog

# ── Structured logging setup ──────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)


# ── Request ID middleware helper ───────────────────────────────────────────────
def new_request_id() -> str:
    return str(uuid.uuid4())


# ── Timing decorator ──────────────────────────────────────────────────────────
def timed(fn: Callable) -> Callable:
    """Log execution time of any function."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        log = get_logger(fn.__qualname__)
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info("executed", latency_ms=round(elapsed_ms, 2))
        return result
    return wrapper


@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """Context-manager timer for blocks."""
    t0 = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - t0) * 1000
    get_logger("timer").info(label, latency_ms=round(elapsed_ms, 2))


# ── LLM Cost Tracker ──────────────────────────────────────────────────────────
@dataclass
class CostTracker:
    """
    Stateful token + cost accumulator.

    Interview talking point:
      "LLM cost scales with volume. I track every call's token usage so
       we can enforce a daily budget, route cheap queries to smaller models
       (e.g. GPT-4o-mini), and report cost-per-alert in dashboards."
    """
    cost_per_1k_input: float = 0.005
    cost_per_1k_output: float = 0.015
    daily_budget_usd: float = 50.0

    total_input_tokens: int = field(default=0)
    total_output_tokens: int = field(default=0)
    call_log: list[dict[str, Any]] = field(default_factory=list)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        cost = (input_tokens / 1000 * self.cost_per_1k_input +
                output_tokens / 1000 * self.cost_per_1k_output)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_log.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
        })
        return cost

    @property
    def total_cost_usd(self) -> float:
        return sum(c["cost_usd"] for c in self.call_log)

    def budget_remaining(self) -> float:
        return self.daily_budget_usd - self.total_cost_usd

    def is_over_budget(self) -> bool:
        return self.total_cost_usd >= self.daily_budget_usd

    def summary(self) -> dict[str, Any]:
        return {
            "total_calls": len(self.call_log),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "budget_remaining_usd": round(self.budget_remaining(), 4),
        }


# Global cost tracker (reset daily in production via a cron/Lambda)
cost_tracker = CostTracker()


# ── JSON-safe serialisation ───────────────────────────────────────────────────
def to_serialisable(obj: Any) -> Any:
    """Recursively convert numpy / pandas types to native Python for JSON."""
    import numpy as np
    import pandas as pd

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serialisable(i) for i in obj]
    return obj

