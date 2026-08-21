"""Rate limiting — in-memory token bucket (Phase 20 §8)."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict


class RateLimiter:
    def __init__(self, max_requests: int = 60, window_s: float = 60.0):
        self.max_requests = max_requests
        self.window_s = window_s
        self._hits: Dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        hits = self._hits[key]
        # Evict outside window
        self._hits[key] = [t for t in hits if now - t < self.window_s]
        if len(self._hits[key]) >= self.max_requests:
            return False
        self._hits[key].append(now)
        return True
