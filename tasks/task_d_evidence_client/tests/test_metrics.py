"""Metrics emission: every call routes through the collector."""
from __future__ import annotations

from unittest.mock import MagicMock

import requests

from evidence_client.http import HttpRunner, RetryPolicy
from evidence_client.metrics import InMemoryMetrics


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)

    def request(self, method, url, **kw):
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


def test_metrics_counts_successful_call():
    metrics = InMemoryMetrics()
    session = _FakeSession([_resp(200, '{"ok":true}')])
    runner = HttpRunner(session=session, metrics=metrics)
    runner.get("http://x", source="pubmed")

    snap = metrics.snapshot()
    assert "pubmed" in snap
    assert snap["pubmed"]["calls_by_outcome"]["ok"] == 1
    assert snap["pubmed"]["calls_by_status"]["200"] == 1
    assert snap["pubmed"]["latency_ms"]["count"] == 1


def test_metrics_distinguishes_http_error_vs_ok():
    metrics = InMemoryMetrics()
    session = _FakeSession([_resp(200), _resp(404), _resp(200)])
    runner = HttpRunner(
        session=session, metrics=metrics,
        retry_policy=RetryPolicy(max_retries=0, base_delay=0.0),
    )
    runner.get("http://x", source="pubmed")
    runner.get("http://x?b", source="pubmed")  # different cache key
    runner.get("http://x?c", source="pubmed")

    snap = metrics.snapshot()
    assert snap["pubmed"]["calls_by_outcome"]["ok"] == 2
    assert snap["pubmed"]["calls_by_outcome"]["http_error"] == 1
    assert snap["pubmed"]["calls_by_status"]["200"] == 2
    assert snap["pubmed"]["calls_by_status"]["404"] == 1


def test_metrics_records_network_error():
    metrics = InMemoryMetrics()
    session = _FakeSession([requests.ConnectionError("dns fail")] * 5)
    runner = HttpRunner(
        session=session, metrics=metrics,
        retry_policy=RetryPolicy(max_retries=1, base_delay=0.0),
    )
    runner.get("http://x", source="faers")
    snap = metrics.snapshot()
    # 2 attempts (initial + 1 retry), both network errors.
    assert snap["faers"]["calls_by_outcome"]["network_error"] == 2


def test_metrics_records_cache_hit():
    metrics = InMemoryMetrics()
    session = _FakeSession([_resp(200, '{"ok":true}')])
    runner = HttpRunner(session=session, metrics=metrics)
    runner.get("http://x", source="pubmed")
    runner.get("http://x", source="pubmed")  # cache hit

    snap = metrics.snapshot()
    assert snap["pubmed"]["calls_by_outcome"].get("cache_hit", 0) == 1


def test_metrics_per_source_attribution():
    metrics = InMemoryMetrics()
    session = _FakeSession([_resp(200), _resp(200), _resp(200)])
    runner = HttpRunner(session=session, metrics=metrics)
    runner.get("http://x", source="pubmed")
    runner.get("http://y", source="faers")
    runner.get("http://z", source="string_db")

    snap = metrics.snapshot()
    assert set(snap.keys()) == {"pubmed", "faers", "string_db"}
    for src in ("pubmed", "faers", "string_db"):
        assert snap[src]["calls_by_outcome"]["ok"] == 1


def test_evidence_client_exposes_metrics(tmp_path):
    """The high-level client exposes a metrics property by default."""
    from evidence_client import EvidenceClient, InMemoryMetrics

    metrics = InMemoryMetrics()
    c = EvidenceClient(metrics=metrics)
    assert c.metrics is metrics
