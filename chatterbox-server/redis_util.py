"""Valkey utilities — connection, rate limiting, caching, pub/sub.

Every function degrades gracefully when Valkey is unavailable so the
application keeps working with in-memory fallbacks.
"""

import json
import logging
import time

import valkey
from config import VALKEY_DB, VALKEY_HOST, VALKEY_PASSWORD, VALKEY_PORT, VALKEY_URL

logger = logging.getLogger(__name__)

# ── Connection ──────────────────────────────────────────────────

_valkey_client: valkey.Valkey | None = None
_available = False


def get_redis() -> valkey.Valkey | None:
    """Return the shared Valkey client (decode_responses=True), or ``None`` if unreachable."""
    global _valkey_client, _available
    if _valkey_client is not None:
        return _valkey_client if _available else None
    try:
        if VALKEY_URL:
            _valkey_client = valkey.from_url(VALKEY_URL, decode_responses=True, socket_timeout=2)
        else:
            kwargs = dict(decode_responses=True, socket_timeout=2)
            if VALKEY_PASSWORD:
                kwargs["password"] = VALKEY_PASSWORD
            _valkey_client = valkey.Valkey(
                host=VALKEY_HOST, port=VALKEY_PORT, db=VALKEY_DB, **kwargs
            )
        _valkey_client.ping()
        _available = True
        logger.info("Valkey connected")
        return _valkey_client
    except valkey.ConnectionError:
        _available = False
        logger.warning("Valkey not available — falling back to in-memory")
        return None


def is_available() -> bool:
    """Return True if Valkey is reachable."""
    get_redis()
    return _available


# ── Rate limiting ───────────────────────────────────────────────

def check_rate_limit(key: str, limit: int, window: int) -> bool:
    """Return True if the request is *within* the limit.

    Uses Valkey INCR + EXPIRE (fixed-window counter).  Falls back to
    ``True`` (allow) when Valkey is down so the app stays usable.

    Args:
        key:    Unique identifier (e.g. ``"rl:ip:1.2.3.4"``).
        limit:  Max requests per window.
        window: Window size in seconds.
    """
    r = get_redis()
    if r is None:
        return True  # no Valkey → allow (in-memory fallback handled by caller)
    try:
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.expire(key, window)
        count, _ = pipe.execute()
        return count <= limit
    except valkey.ValkeyError as e:
        logger.warning(f"Valkey rate limit error: {e}")
        return True


# ── Caching ─────────────────────────────────────────────────────

def cache_get(key: str) -> str | None:
    """Get a cached value. Returns None on miss or Valkey failure."""
    r = get_redis()
    if r is None:
        return None
    try:
        return r.get(key)
    except valkey.ValkeyError:
        return None


def cache_set(key: str, value: str, ttl: int = 300):
    """Set a cached value with TTL (seconds). No-op when Valkey is down."""
    r = get_redis()
    if r is None:
        return
    try:
        r.set(key, value, ex=ttl)
    except valkey.ValkeyError:
        pass


# ── Pub/Sub (job status) ───────────────────────────────────────

JOB_STATUS_CHANNEL = "job:status"


def publish_job_status(access_code: str, status: str, **extra):
    """Publish a job status change to the ``job:status`` channel.

    Message format (JSON)::

        {"access_code": "...", "status": "...", ...extra_fields...}
    """
    r = get_redis()
    if r is None:
        return
    try:
        payload = {"access_code": access_code, "status": status, **extra}
        r.publish(JOB_STATUS_CHANNEL, json.dumps(payload))
    except valkey.ValkeyError as e:
        logger.warning(f"Valkey publish error: {e}")


# ── In-memory rate limiter (fallback) ───────────────────────────

class InMemoryRateLimiter:
    """Sliding-window rate limiter using a plain dict.

    Used when Valkey is unavailable.  One instance per limiter policy.
    """

    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        self._store: dict[str, list[float]] = {}

    def check(self, key: str) -> bool:
        """Return True if the request is within the limit."""
        now = time.time()
        window_start = now - self.window
        timestamps = self._store.get(key, [])
        pruned = [t for t in timestamps if t > window_start]
        if pruned:
            self._store[key] = pruned
        elif key in self._store:
            del self._store[key]
        if len(pruned) >= self.limit:
            return False
        self._store.setdefault(key, pruned).append(now)
        return True
