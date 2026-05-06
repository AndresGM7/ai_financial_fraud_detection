"""
Utility functions: logging setup and common helpers.
"""
import logging
import sys
from typing import Any


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a consistent format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def safe_get(d: dict, key: str, default: Any = None) -> Any:
    """Safely retrieve a value from a dict."""
    return d.get(key, default)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float value between lo and hi."""
    return max(lo, min(hi, value))
