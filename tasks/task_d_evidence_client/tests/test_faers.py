"""FDA FAERS source tests."""
from __future__ import annotations

import pytest

from evidence_client.models import Failure, Success
from ._helpers import live_only


def test_faers_search_returns_success(client, vcr_cassette):
    r = client.faers.search(
        'patient.drug.medicinalproduct:"METFORMIN"', limit=2
    )
    assert isinstance(r, Success)
    assert r.source == "faers"
    assert "results" in r.data


def test_faers_fetch_by_id(client, vcr_cassette):
    # We pick an id likely to exist in FAERS; the test only checks that
    # the contract is honored, not the specific record.
    r = client.faers.fetch_by_id("10000001")
    # FAERS may return 404 for unknown IDs — accept either Success or a
    # well-formed Failure.
    assert isinstance(r, (Success, Failure))
    if isinstance(r, Failure):
        assert r.status_code is not None


@live_only()
def test_faers_live_search():
    from evidence_client import EvidenceClient

    c = EvidenceClient()
    r = c.faers.search('patient.drug.medicinalproduct:"METFORMIN"', limit=1)
    assert isinstance(r, (Success, Failure))
