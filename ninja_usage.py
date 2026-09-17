"""Process-local, per-key estimates of Ninja usage; never store the raw token."""

from collections import deque
from email.utils import parsedate_to_datetime
from hashlib import sha256
import math
from threading import Lock
import time

HOURLY_LIMIT = 50
_LOCK = Lock()
_USAGE = {}


def _entry(token, now):
    fingerprint = sha256(token.encode()).hexdigest()
    entry = _USAGE.setdefault(fingerprint, {"calls": deque(), "retry_until": 0.0})
    while entry["calls"] and entry["calls"][0] <= now - 3600:
        entry["calls"].popleft()
    return entry


def record_request(token):
    """Count every outgoing attempt conservatively, including retries and failures."""
    with _LOCK:
        now = time.time()
        _entry(token, now)["calls"].append(now)


def record_response(token, response):
    if response.status_code != 429:
        return
    value = response.headers.get("Retry-After")
    if not value:
        return
    now = time.time()
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - now
        except (TypeError, ValueError, OverflowError):
            return
    if not math.isfinite(seconds):
        return
    with _LOCK:
        entry = _entry(token, now)
        entry["retry_until"] = max(entry["retry_until"], now + max(0, seconds))


def usage_snapshot(token):
    with _LOCK:
        now = time.time()
        entry = _entry(token, now)
        calls = entry["calls"]
        return {
            "used": len(calls), "remaining": max(0, HOURLY_LIMIT - len(calls)),
            "limit": HOURLY_LIMIT,
            "next_release_seconds": math.ceil(max(0, calls[0] + 3600 - now)) if calls else 0,
            "retry_seconds": math.ceil(max(0, entry["retry_until"] - now)),
        }
