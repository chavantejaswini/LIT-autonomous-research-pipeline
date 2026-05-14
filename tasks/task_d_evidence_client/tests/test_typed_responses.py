"""Per-source response schemas catch upstream shape changes at the boundary.

When the API returns something unexpected, the source surfaces a clean
`Failure(message="parse error: …")` instead of letting a `KeyError`
escape from caller code.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import requests

from evidence_client import EvidenceClient
from evidence_client.http import HttpRunner
from evidence_client.metrics import NullMetrics
from evidence_client.models import Failure
from evidence_client.response_types import (
    PubMedSearchResponse,
    StringDbResponse,
    parse_json_into,
)


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)

    def request(self, method, url, **kw):
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _resp(status, body):
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.text = body
    r.headers = {}
    r.url = "http://test"
    return r


def test_pubmed_schema_accepts_valid_payload():
    parser = parse_json_into(PubMedSearchResponse)
    parsed = parser('{"esearchresult": {"idlist": ["123", "456"]}}')
    assert parsed["esearchresult"]["idlist"] == ["123", "456"]


def test_pubmed_schema_rejects_missing_top_level_key():
    parser = parse_json_into(PubMedSearchResponse)
    try:
        parser('{"some_other_key": {}}')
    except Exception as exc:
        # pydantic raises ValidationError — we just care it's caught.
        assert "esearchresult" in str(exc) or "validation" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("expected validation to fail")


def test_string_db_unwraps_top_level_list():
    parser = parse_json_into(StringDbResponse)
    parsed = parser('[{"preferredName": "TP53"}]')
    assert parsed["results"] == [{"preferredName": "TP53"}]


def test_malformed_upstream_surfaces_as_failure():
    """End-to-end: a real-looking source call gets garbage back, returns Failure."""
    session = _FakeSession([_resp(200, '{"completely_wrong_shape": 42}')])
    runner = HttpRunner(session=session, metrics=NullMetrics())
    # Construct a client whose runner is mocked.
    client = EvidenceClient()
    client._runner = runner  # type: ignore
    client.pubmed._runner = runner  # type: ignore

    result = client.pubmed.search("anything")
    assert isinstance(result, Failure)
    assert "parse error" in result.message
