"""Shared HTTP infrastructure: retry policy, TTL cache, structured logging,
circuit breaker, metrics.

All source clients route through `HttpRunner` to share:
  * configurable timeout (default 30s)
  * exponential backoff on 429 *and* 5xx, max 3 retries, honoring Retry-After
  * structured per-call log line with latency + outcome
  * **thread-safe** in-memory TTL cache keyed by (method, url, params, body)
  * optional per-source circuit breaker — fail fast for known-dead sources
  * optional metrics collector — counters + latency histogram
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from .circuit_breaker import CircuitBreaker, CircuitOpenError
from .metrics import MetricsCollector, NullMetrics

logger = logging.getLogger("evidence_client.http")

# Status codes worth retrying. 429 = rate limited; 502/503/504 are transient
# upstream failures (bad gateway, service unavailable, gateway timeout).
DEFAULT_RETRY_STATUSES = frozenset({429, 502, 503, 504})


@dataclass
class RetryPolicy:
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0
    retry_on_status: frozenset[int] = field(default_factory=lambda: DEFAULT_RETRY_STATUSES)


class TTLCache:
    """Bounded in-memory TTL cache. Safe to share across threads."""

    def __init__(self, ttl_seconds: float = 300.0, max_entries: int = 256) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            record = self._store.get(key)
            if record is None:
                return None
            expires_at, value = record
            if expires_at < time.monotonic():
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._store) >= self.max_entries:
                # Evict the entry closest to expiry.
                oldest = min(self._store, key=lambda k: self._store[k][0])
                del self._store[oldest]
            self._store[key] = (time.monotonic() + self.ttl_seconds, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


def cache_key(method: str, url: str, params: dict | None, body: Any) -> str:
    payload = json.dumps(
        {"m": method.upper(), "u": url, "p": params or {}, "b": body},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class HttpResponse:
    status_code: int
    headers: dict[str, str]
    text: str
    url: str

    def json(self) -> Any:
        return json.loads(self.text)


class HttpRunner:
    """Wraps `requests` with retry + cache + logging + breaker + metrics.

    Returns `HttpResponse` on every HTTP-level outcome (including 4xx/5xx),
    or `RuntimeError` only when retries are exhausted on network errors or
    the circuit breaker rejects the call.

    Parameters
    ----------
    session : requests.Session | None
        Shared session. Defaults to a fresh one with default pooling.
    timeout : float
        Per-call timeout in seconds.
    retry_policy : RetryPolicy | None
        Retry behavior — default backs off 429/502/503/504 with `Retry-After`.
    cache : TTLCache | None
        In-memory TTL cache; default 5 min / 256 entries.
    circuit_breaker : CircuitBreaker | None
        Optional per-source breaker. If provided, calls to a source whose
        breaker is OPEN return immediately as `Failure(message="circuit-open")`.
    metrics : MetricsCollector | None
        Optional metrics sink for counters + latency. Defaults to a no-op.
    """

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        cache: TTLCache | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._session = session if session is not None else requests.Session()
        self._timeout = timeout
        self._retry = retry_policy if retry_policy is not None else RetryPolicy()
        self._cache = cache if cache is not None else TTLCache()
        self._breaker = circuit_breaker
        self._metrics = metrics if metrics is not None else NullMetrics()

    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics

    @property
    def breaker(self) -> CircuitBreaker | None:
        return self._breaker

    def get(
        self,
        url: str,
        *,
        source: str = "unknown",
        params: dict | None = None,
        headers: dict | None = None,
    ) -> HttpResponse | RuntimeError | CircuitOpenError:
        return self._call(
            "GET", url, source=source, params=params, headers=headers, body=None
        )

    def post(
        self,
        url: str,
        *,
        source: str = "unknown",
        params: dict | None = None,
        json_body: Any = None,
        headers: dict | None = None,
    ) -> HttpResponse | RuntimeError | CircuitOpenError:
        return self._call(
            "POST", url, source=source, params=params, headers=headers, body=json_body
        )

    def _call(
        self,
        method: str,
        url: str,
        *,
        source: str,
        params: dict | None,
        headers: dict | None,
        body: Any,
    ) -> HttpResponse | RuntimeError | CircuitOpenError:
        # Circuit breaker: fail fast if the source is currently tripped.
        if self._breaker is not None and not self._breaker.allow_request(source):
            self._metrics.record(
                source=source, status_code=None, latency_ms=0.0, outcome="circuit_open"
            )
            return CircuitOpenError(f"circuit open for source={source}")

        key = cache_key(method, url, params, body)
        cached = self._cache.get(key)
        if cached is not None:
            logger.info("http.cache_hit source=%s method=%s url=%s", source, method, url)
            self._metrics.record(
                source=source, status_code=cached.status_code,
                latency_ms=0.0, outcome="cache_hit",
            )
            return cached

        attempt = 0
        last_error: str = ""
        while attempt <= self._retry.max_retries:
            attempt += 1
            t0 = time.perf_counter()
            try:
                resp = self._session.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    json=body if body is not None else None,
                    timeout=self._timeout,
                )
            except requests.RequestException as exc:
                latency_ms = (time.perf_counter() - t0) * 1000
                logger.warning(
                    "http.exception source=%s method=%s url=%s attempt=%d latency_ms=%.1f error=%r",
                    source, method, url, attempt, latency_ms, exc,
                )
                self._metrics.record(
                    source=source, status_code=None,
                    latency_ms=latency_ms, outcome="network_error",
                )
                last_error = str(exc)
                if attempt > self._retry.max_retries:
                    if self._breaker is not None:
                        self._breaker.record_failure(source)
                    return RuntimeError(last_error)
                self._sleep_backoff(attempt)
                continue

            latency_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "http.call source=%s method=%s url=%s attempt=%d status=%d latency_ms=%.1f",
                source, method, url, attempt, resp.status_code, latency_ms,
            )
            self._metrics.record(
                source=source, status_code=resp.status_code,
                latency_ms=latency_ms,
                outcome="ok" if 200 <= resp.status_code < 300 else "http_error",
            )

            if resp.status_code in self._retry.retry_on_status and attempt <= self._retry.max_retries:
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                delay = retry_after if retry_after is not None else self._backoff_delay(attempt)
                logger.warning(
                    "http.retryable source=%s url=%s status=%d sleep_s=%.2f attempt=%d",
                    source, url, resp.status_code, delay, attempt,
                )
                time.sleep(min(delay, self._retry.max_delay))
                continue

            out = HttpResponse(
                status_code=resp.status_code,
                headers=dict(resp.headers),
                text=resp.text,
                url=resp.url,
            )
            if 200 <= resp.status_code < 300:
                self._cache.set(key, out)
                if self._breaker is not None:
                    self._breaker.record_success(source)
            else:
                if self._breaker is not None:
                    self._breaker.record_failure(source)
            return out

        if self._breaker is not None:
            self._breaker.record_failure(source)
        return RuntimeError(last_error or "retries exhausted")

    def _sleep_backoff(self, attempt: int) -> None:
        time.sleep(self._backoff_delay(attempt))

    def _backoff_delay(self, attempt: int) -> float:
        return min(self._retry.base_delay * (2 ** (attempt - 1)), self._retry.max_delay)


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a `Retry-After` header — either delta-seconds or HTTP-date."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        from datetime import datetime, timezone

        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (dt - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError):
        return None
