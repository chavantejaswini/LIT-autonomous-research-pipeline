"""ClinicalTrials.gov v2 source tests."""
from __future__ import annotations

import pytest

from evidence_client.models import Failure, Success
from ._helpers import live_only


def test_ctg_search_returns_success(client, vcr_cassette):
    r = client.clinical_trials.search("rapamycin healthspan", page_size=2)
    assert isinstance(r, Success)
    assert r.source == "clinical_trials"
    assert "studies" in r.data or "totalCount" in r.data


def test_ctg_fetch_by_id_returns_success(client, vcr_cassette):
    r = client.clinical_trials.fetch_by_id("NCT04566393")
    assert isinstance(r, Success)
    assert r.source == "clinical_trials"


@live_only()
def test_ctg_live_search():
    from evidence_client import EvidenceClient

    c = EvidenceClient()
    r = c.clinical_trials.search("aging", page_size=1)
    assert r.ok()
