"""Redis utilities — connection, rate limiting, caching, pub/sub.

Every function degrades gracefully when Redis is unavailable so the
application keeps working with in-memory fallbacks.
"""

import json
import logging
import secrets
import time
from functools import wraps

import redis

from config import REDIS_URL

logger = logging.getLogger(__name__)

# ── Connection ──────────────────────────────────────────────────

_redis_client: redis.Redis | None = None
_session_redis_client: redis.Redis | None = None
_available = False


def get_redis() -> redis.Redis | None:
    """Return the shared Redis client (decode_responses=True), or ``None`` if unreachable."""
    global _redis_client, _available
    if _redis_client is not None:
        return _redis_client if _available else None
    try:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
        _redis_client.ping()
        _available = True
        logger.info(f"Redis connected: {REDIS_URL}")
        return _redis_client
    except redis.ConnectionError:
        _available = False
        logger.warning(f"Redis not available at {REDIS_URL} — falling back to in-memory")
        return None


def get_session_redis() -> redis.Redis | None:
    """Return a Redis client for Flask-Session (binary responses, no decode).

    Flask-Session stores data as binary msgpack, so it needs
    ``decode_responses=False`` (the default).  This client must NOT be
    shared with application code that expects decoded strings.
    """
    global _session_redis_client
    if _available is False and _redis_client is None:
        return None
    if _session_redis_client is not None:
        return _session_redis_client
    try:
        _session_redis_client = redis.from_url(REDIS_URL, socket_timeout=2)
        _session_redis_client.ping()
        return _session_redis_client
    except redis.ConnectionError:
        return None


def is_available() -> bool:
    """Return True if Redis is reachable."""
    get_redis()
    return _available


# ── Rate limiting ───────────────────────────────────────────────

def check_rate_limit(key: str, limit: int, window: int) -> bool:
    """Return True if the request is *within* the limit.

    Uses Redis INCR + EXPIRE (fixed-window counter).  Falls back to
    ``True`` (allow) when Redis is down so the app stays usable.

    Args:
        key:    Unique identifier (e.g. ``"rl:ip:1.2.3.4"``).
        limit:  Max requests per window.
        window: Window size in seconds.
    """
    r = get_redis()
    if r is None:
        return True  # no Redis → allow (in-memory fallback handled by caller)
    try:
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.expire(key, window)
        count, _ = pipe.execute()
        return count <= limit
    except redis.RedisError as e:
        logger.warning(f"Redis rate limit error: {e}")
        return True


# ── Caching ─────────────────────────────────────────────────────

def cache_get(key: str) -> str | None:
    """Get a cached value. Returns None on miss or Redis failure."""
    r = get_redis()
    if r is None:
        return None
    try:
        return r.get(key)
    except redis.RedisError:
        return None


def cache_set(key: str, value: str, ttl: int = 300):
    """Set a cached value with TTL (seconds). No-op when Redis is down."""
    r = get_redis()
    if r is None:
        return
    try:
        r.set(key, value, ex=ttl)
    except redis.RedisError:
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
    except redis.RedisError as e:
        logger.warning(f"Redis publish error: {e}")


# ── In-memory rate limiter (fallback) ───────────────────────────

class InMemoryRateLimiter:
    """Sliding-window rate limiter using a plain dict.

    Used when Redis is unavailable.  One instance per limiter policy.
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
