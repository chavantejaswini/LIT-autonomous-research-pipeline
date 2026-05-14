"""Concurrent calls to the HTTP runner must be safe.

The TTL cache is the natural concurrent-write point. With many threads
issuing the same GET, the cache must serve everyone correctly without
deadlocking or losing entries.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import requests

from evidence_client.http import HttpRunner, RetryPolicy, TTLCache
from evidence_client.metrics import InMemoryMetrics


def _resp(status=200, body='{"ok":true}'):
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.text = body
    r.headers = {}
    r.url = "http://test"
    return r


class _ConcurrentSafeSession:
    """Returns 200 for every call. Thread-safe via internal lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0

    def request(self, method, url, **kw):
        with self._lock:
            self.calls += 1
        return _resp(200, f'{{"call":{self.calls}}}')


def test_cache_get_set_is_thread_safe():
    cache = TTLCache(ttl_seconds=10.0, max_entries=1000)

    def writer(n):
        for i in range(n):
            cache.set(f"k{i}", i)
            cache.get(f"k{i}")

    threads = [threading.Thread(target=writer, args=(200,)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # No deadlock + every key reachable.
    assert len(cache) <= 1000  # bounded
    # All threads agreed on cached values for at least the recent keys.
    assert cache.get("k199") == 199


def test_runner_concurrent_calls_do_not_crash():
    session = _ConcurrentSafeSession()
    metrics = InMemoryMetrics()
    runner = HttpRunner(
        session=session, metrics=metrics,
        retry_policy=RetryPolicy(max_retries=0, base_delay=0.0),
    )

    def worker(i):
        for j in range(20):
            runner.get(f"http://x?{i}-{j}", source="pubmed")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = metrics.snapshot()
    # Each call attributed to pubmed; no exceptions propagated.
    assert snap["pubmed"]["calls_by_outcome"]["ok"] == 8 * 20


def test_circuit_breaker_concurrent_failures_open_once():
    """Many threads simultaneously recording failures must converge to OPEN."""
    from evidence_client.circuit_breaker import CircuitBreaker, State

    cb = CircuitBreaker(failure_threshold=10, recovery_seconds=60.0)

    def fail_a_few():
        for _ in range(5):
            cb.record_failure("pubmed")

    threads = [threading.Thread(target=fail_a_few) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 50 total failures recorded — well over threshold.
    assert cb.state("pubmed") is State.OPEN
