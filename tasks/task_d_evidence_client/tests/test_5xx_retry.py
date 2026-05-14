"""Transient 5xx errors must be retried just like 429."""
from __future__ import annotations

from unittest.mock import MagicMock

import requests

from evidence_client.http import HttpResponse, HttpRunner, RetryPolicy


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def request(self, method, url, **kw):
        self.calls.append((method, url))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _resp(status, body="{}", headers=None):
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.text = body
    r.headers = headers or {}
    r.url = "http://test"
    return r


def test_retries_on_502_then_succeeds():
    session = _FakeSession(
        [_resp(502, "bad gateway"), _resp(503, "service unavailable"), _resp(200, '{"ok":true}')]
    )
    runner = HttpRunner(
        session=session,
        retry_policy=RetryPolicy(max_retries=3, base_delay=0.0),
    )
    out = runner.get("http://x", source="test")
    assert isinstance(out, HttpResponse) and out.status_code == 200
    assert len(session.calls) == 3


def test_retries_on_504_then_gives_up():
    session = _FakeSession([_resp(504, "gateway timeout")] * 10)
    runner = HttpRunner(
        session=session,
        retry_policy=RetryPolicy(max_retries=2, base_delay=0.0),
    )
    out = runner.get("http://x", source="test")
    # After exhausting retries, the last 504 surfaces as an HttpResponse.
    assert isinstance(out, HttpResponse) and out.status_code == 504


def test_does_not_retry_on_500_by_default():
    """500 is not in DEFAULT_RETRY_STATUSES — it's a real server bug, not transient."""
    session = _FakeSession([_resp(500, "internal error")])
    runner = HttpRunner(
        session=session,
        retry_policy=RetryPolicy(max_retries=3, base_delay=0.0),
    )
    out = runner.get("http://x", source="test")
    assert isinstance(out, HttpResponse) and out.status_code == 500
    assert len(session.calls) == 1  # exactly one — no retry


def test_retry_policy_is_configurable():
    """A custom RetryPolicy can extend the retry set to include 500."""
    session = _FakeSession([_resp(500, ""), _resp(200, '{"ok":true}')])
    runner = HttpRunner(
        session=session,
        retry_policy=RetryPolicy(
            max_retries=3,
            base_delay=0.0,
            retry_on_status=frozenset({500, 502, 503, 504, 429}),
        ),
    )
    out = runner.get("http://x", source="test")
    assert isinstance(out, HttpResponse) and out.status_code == 200
    assert len(session.calls) == 2
