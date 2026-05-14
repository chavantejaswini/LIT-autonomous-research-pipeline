"""Tests for the shared HTTP layer: retry, cache, 429 backoff, failure mapping."""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from evidence_client.http import (
    HttpResponse,
    HttpRunner,
    RetryPolicy,
    TTLCache,
    _parse_retry_after,
    cache_key,
)
from evidence_client.models import Failure, Success
from evidence_client.sources._base import call_to_result


class _FakeSession:
    """Tiny `requests.Session` stand-in driven by a list of canned responses."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def request(self, method, url, params=None, headers=None, json=None, timeout=None):
        self.calls.append((method, url))
        if not self._responses:
            raise RuntimeError("no more canned responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _resp(status: int, body: str = "{}", headers: dict | None = None):
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.text = body
    r.headers = headers or {}
    r.url = "http://test"
    return r


def test_cache_returns_stored_value_within_ttl():
    cache = TTLCache(ttl_seconds=10)
    cache.set("k", "v")
    assert cache.get("k") == "v"


def test_cache_expires_after_ttl(monkeypatch):
    cache = TTLCache(ttl_seconds=0.01)
    cache.set("k", "v")
    time.sleep(0.05)
    assert cache.get("k") is None


def test_cache_key_is_stable_across_arg_order():
    a = cache_key("GET", "u", {"b": 2, "a": 1}, None)
    b = cache_key("GET", "u", {"a": 1, "b": 2}, None)
    assert a == b


def test_runner_caches_2xx_response_and_skips_second_call():
    session = _FakeSession([_resp(200, '{"x":1}')])
    runner = HttpRunner(session=session)
    a = runner.get("http://x")
    b = runner.get("http://x")
    assert isinstance(a, HttpResponse) and a.status_code == 200
    assert isinstance(b, HttpResponse) and b.status_code == 200
    assert len(session.calls) == 1  # second call served from cache


def test_runner_does_not_cache_4xx_response():
    session = _FakeSession([_resp(404, "not found"), _resp(404, "still not found")])
    runner = HttpRunner(session=session)
    a = runner.get("http://x")
    b = runner.get("http://x")
    assert isinstance(a, HttpResponse) and a.status_code == 404
    assert isinstance(b, HttpResponse) and b.status_code == 404
    assert len(session.calls) == 2


def test_runner_retries_on_429_and_succeeds():
    session = _FakeSession(
        [
            _resp(429, "", {"Retry-After": "0"}),
            _resp(429, "", {"Retry-After": "0"}),
            _resp(200, '{"ok":true}'),
        ]
    )
    runner = HttpRunner(
        session=session, retry_policy=RetryPolicy(max_retries=3, base_delay=0.0)
    )
    out = runner.get("http://x")
    assert isinstance(out, HttpResponse) and out.status_code == 200
    assert len(session.calls) == 3


def test_runner_gives_up_after_max_retries_on_429():
    session = _FakeSession(
        [_resp(429, "", {"Retry-After": "0"})] * 5
    )
    runner = HttpRunner(
        session=session, retry_policy=RetryPolicy(max_retries=3, base_delay=0.0)
    )
    out = runner.get("http://x")
    # Eventually returns the last 429 wrapped as HttpResponse.
    assert isinstance(out, HttpResponse) and out.status_code == 429


def test_runner_returns_runtime_error_after_network_exhaustion():
    session = _FakeSession(
        [requests.ConnectionError("boom")] * 5
    )
    runner = HttpRunner(
        session=session, retry_policy=RetryPolicy(max_retries=2, base_delay=0.0)
    )
    out = runner.get("http://x")
    assert isinstance(out, RuntimeError)


def test_call_to_result_maps_4xx_to_failure():
    session = _FakeSession([_resp(404, "missing")])
    runner = HttpRunner(session=session)
    r = call_to_result(
        runner, "GET", "http://x", source="test", query_id="q1"
    )
    assert isinstance(r, Failure)
    assert r.status_code == 404


def test_call_to_result_maps_parse_failure_to_failure():
    session = _FakeSession([_resp(200, "not json {{{")])
    runner = HttpRunner(session=session)
    r = call_to_result(
        runner, "GET", "http://x", source="test", query_id="q2"
    )
    assert isinstance(r, Failure)
    assert "parse error" in r.message


def test_call_to_result_maps_network_failure_to_failure():
    session = _FakeSession([requests.ConnectionError("dns fail")] * 5)
    runner = HttpRunner(
        session=session, retry_policy=RetryPolicy(max_retries=1, base_delay=0.0)
    )
    r = call_to_result(
        runner, "GET", "http://x", source="test", query_id="q3"
    )
    assert isinstance(r, Failure)
    assert "network error" in r.message


def test_parse_retry_after_handles_seconds_and_date():
    assert _parse_retry_after("5") == 5.0
    assert _parse_retry_after(None) is None
    # HTTP-date form should not raise (could be past => 0).
    result = _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT")
    assert result is None or result >= 0.0
