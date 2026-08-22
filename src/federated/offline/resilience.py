"""Offline-first endpoint resilience — bounded retry with backoff (Enhancement 10)."""

from __future__ import annotations

import time
from typing import Callable


def with_backoff(fn: Callable, max_retries: int = 3, base: float = 1.0):
    """Execute fn with bounded retry and exponential backoff. No infinite loops."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(base * (2 ** attempt))
