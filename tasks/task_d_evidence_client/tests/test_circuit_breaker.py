"""Circuit-breaker behavior: opens after N failures, recovers via HALF_OPEN.

The breaker is exercised through the `HttpRunner` so we also verify that
short-circuited calls return `CircuitOpenError` (which the source layer
wraps in a `Failure`).
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import requests

from evidence_client.circuit_breaker import CircuitBreaker, CircuitOpenError
from evidence_client.circuit_breaker import State
from evidence_client.http import HttpResponse, HttpRunner, RetryPolicy


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def request(self, method, url, **kw):
        self.calls += 1
        if not self._responses:
            raise RuntimeError("no more canned responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _resp(status, body="{}"):
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.text = body
    r.headers = {}
    r.url = "http://test"
    return r


def test_breaker_opens_after_consecutive_failures():
    cb = CircuitBreaker(failure_threshold=3, recovery_seconds=10.0)
    for _ in range(3):
        cb.record_failure("pubmed")
    assert cb.state("pubmed") is State.OPEN
    assert cb.allow_request("pubmed") is False


def test_breaker_stays_closed_on_success_streak():
    cb = CircuitBreaker(failure_threshold=3, recovery_seconds=10.0)
    cb.record_failure("pubmed")
    cb.record_failure("pubmed")
    cb.record_success("pubmed")  # resets consecutive count
    cb.record_failure("pubmed")
    cb.record_failure("pubmed")
    # Still 2 consecutive failures since the last success — should be CLOSED.
    assert cb.state("pubmed") is State.CLOSED


def test_breaker_per_source_isolation():
    cb = CircuitBreaker(failure_threshold=2, recovery_seconds=10.0)
    cb.record_failure("pubmed")
    cb.record_failure("pubmed")
    # PubMed is open; FAERS untouched.
    assert cb.state("pubmed") is State.OPEN
    assert cb.state("faers") is State.CLOSED


def test_breaker_half_opens_after_recovery_window(monkeypatch):
    cb = CircuitBreaker(failure_threshold=2, recovery_seconds=0.05)
    cb.record_failure("pubmed")
    cb.record_failure("pubmed")
    assert cb.state("pubmed") is State.OPEN
    time.sleep(0.06)
    # Next call probes: HALF_OPEN, allow_request returns True.
    assert cb.allow_request("pubmed") is True
    assert cb.state("pubmed") is State.HALF_OPEN
    # Probe success → CLOSED.
    cb.record_success("pubmed")
    assert cb.state("pubmed") is State.CLOSED


def test_breaker_reopens_on_half_open_failure():
    cb = CircuitBreaker(failure_threshold=2, recovery_seconds=0.01)
    cb.record_failure("pubmed")
    cb.record_failure("pubmed")
    time.sleep(0.02)
    cb.allow_request("pubmed")  # transition to HALF_OPEN
    cb.record_failure("pubmed")
    assert cb.state("pubmed") is State.OPEN


def test_runner_short_circuits_when_breaker_open():
    cb = CircuitBreaker(failure_threshold=2, recovery_seconds=30.0)
    cb.record_failure("pubmed")
    cb.record_failure("pubmed")

    # The fake session has a 200 ready, but the breaker should prevent the
    # call entirely. `session.calls` must stay 0.
    session = _FakeSession([_resp(200)])
    runner = HttpRunner(
        session=session,
        circuit_breaker=cb,
        retry_policy=RetryPolicy(max_retries=0, base_delay=0.0),
    )
    out = runner.get("http://x", source="pubmed")
    assert isinstance(out, CircuitOpenError)
    assert session.calls == 0


def test_runner_records_failure_on_5xx_for_breaker():
    """The breaker should see persistent 5xx errors and open the circuit."""
    cb = CircuitBreaker(failure_threshold=2, recovery_seconds=30.0)
    # Two failed attempts (each returns 500 — not retried by default).
    session = _FakeSession([_resp(500), _resp(500)])
    runner = HttpRunner(
        session=session,
        circuit_breaker=cb,
        retry_policy=RetryPolicy(max_retries=0, base_delay=0.0),
    )
    runner.get("http://x", source="pubmed")
    runner.get("http://x", source="pubmed")
    assert cb.state("pubmed") is State.OPEN


def test_runner_short_circuits_surfaces_as_failure_at_source_layer():
    """The source layer wraps `CircuitOpenError` in a `Failure`."""
    from evidence_client.models import Failure
    from evidence_client.sources._base import call_to_result

    cb = CircuitBreaker(failure_threshold=1, recovery_seconds=30.0)
    cb.record_failure("pubmed")
    session = _FakeSession([])
    runner = HttpRunner(session=session, circuit_breaker=cb)
    result = call_to_result(
        runner, "GET", "http://x", source="pubmed", query_id="q1"
    )
    assert isinstance(result, Failure)
    assert "circuit-open" in result.message
