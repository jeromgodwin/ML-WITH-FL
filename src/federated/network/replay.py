"""Replay protection — rejects stale or repeated model-update submissions (Phase 19).

Uses round identifiers, model version, and request identifiers (or equivalent) to
reject obviously stale or repeated submissions.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Set, Tuple


class ReplayProtection:
    """In-memory replay protection with TTL.

    Tracks (client_id, request_id) and (client_id, round, model_version) to
    reject repeats and stale rounds. Entries expire after ttl_s.
    """

    def __init__(self, ttl_s: float = 3600.0, max_entries: int = 10000):
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        # request_id -> expiry
        self._seen_requests: Dict[str, float] = {}
        # (client_id, round, model_version) -> expiry
        self._seen_rounds: Dict[Tuple[str, int, str], float] = {}
        # Highest round seen per client (for stale detection)
        self._highest_round: Dict[str, int] = {}

    def _evict_expired(self) -> None:
        now = time.time()
        self._seen_requests = {k: v for k, v in self._seen_requests.items() if v > now}
        self._seen_rounds = {k: v for k, v in self._seen_rounds.items() if v > now}
        # Bound size
        if len(self._seen_requests) > self.max_entries:
            # Remove oldest
            oldest = sorted(self._seen_requests, key=lambda k: self._seen_requests[k])[: len(self._seen_requests) - self.max_entries]
            for k in oldest:
                del self._seen_requests[k]

    def check_and_record(
        self,
        client_id: str,
        request_id: Optional[str] = None,
        round_number: Optional[int] = None,
        model_version: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Check if the submission is fresh. Records it if accepted.

        Returns (accepted, reason). accepted=False means reject as replay/stale.
        """
        self._evict_expired()
        now = time.time()
        expiry = now + self.ttl_s

        # 1. Request ID replay check
        if request_id is not None:
            key = f"{client_id}:{request_id}"
            if key in self._seen_requests:
                return False, f"replayed request_id {request_id} for {client_id}"
            self._seen_requests[key] = expiry

        # 2. Round+version replay check
        if round_number is not None and model_version is not None:
            rkey = (client_id, int(round_number), str(model_version))
            if rkey in self._seen_rounds:
                return False, f"replayed round {round_number} version {model_version} for {client_id}"
            self._seen_rounds[rkey] = expiry

        # 3. Stale round check — reject if round < highest seen for this client
        if round_number is not None:
            highest = self._highest_round.get(client_id, -1)
            if int(round_number) < highest:
                return False, f"stale round {round_number} < highest {highest} for {client_id}"
            if int(round_number) > highest:
                self._highest_round[client_id] = int(round_number)

        return True, "accepted"

    def is_replay(
        self,
        client_id: str,
        request_id: Optional[str] = None,
        round_number: Optional[int] = None,
        model_version: Optional[str] = None,
    ) -> bool:
        """Non-recording check (for testing)."""
        if request_id is not None and f"{client_id}:{request_id}" in self._seen_requests:
            return True
        if round_number is not None and model_version is not None:
            if (client_id, int(round_number), str(model_version)) in self._seen_rounds:
                return True
        return False
