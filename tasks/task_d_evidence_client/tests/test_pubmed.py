"""PubMed source tests (VCR-replay)."""
from __future__ import annotations

import pytest

from evidence_client.models import Failure, Success
from ._helpers import live_only


def test_pubmed_search_returns_success(client, vcr_cassette):
    result = client.pubmed.search("CRISPR aging", retmax=3)
    assert isinstance(result, Success)
    assert result.source == "pubmed"
    assert "esearchresult" in result.data


def test_pubmed_fetch_by_id_returns_success(client, vcr_cassette):
    result = client.pubmed.fetch_by_id("23456789")
    assert isinstance(result, Success)
    assert result.source == "pubmed"


@live_only()
def test_pubmed_live_search():
    """Live integration — runs only when EVIDENCE_LIVE=1."""
    from evidence_client import EvidenceClient

    c = EvidenceClient()
    r = c.pubmed.search("longevity", retmax=2)
    assert r.ok()
