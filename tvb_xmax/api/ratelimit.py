"""Token-bucket rate limiting per account + tier.

Uses an in-memory dict of (username, endpoint) -> (tokens, last_refill).
For multi-worker deployments, swap for Redis (same interface).
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

from .auth import TIER_LIMITS

_buckets: dict = defaultdict(lambda: [0.0, time.time()])
_lock = Lock()


def check(username: str, tier: str, endpoint: str, batch: int = 1) -> tuple:
    """Return (allowed, retry_after_seconds, remaining).

    Refills tokens at ``rate_per_min/60`` per second up to the bucket cap.
    Each request costs ``batch`` tokens (so big batches cost more).
    """
    rate_per_min, max_batch = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
    if batch > max_batch:
        return False, 0.0, 0, f"batch {batch} exceeds tier {tier} max {max_batch}"
    cap = rate_per_min
    rate = rate_per_min / 60.0
    key = (username, endpoint)
    now = time.time()
    with _lock:
        tokens, last = _buckets[key]
        tokens = min(cap, tokens + (now - last) * rate)
        if tokens < batch:
            need = batch - tokens
            retry = need / rate
            _buckets[key] = [tokens, now]
            return False, retry, int(tokens), None
        tokens -= batch
        _buckets[key] = [tokens, now]
        return True, 0.0, int(tokens), None
