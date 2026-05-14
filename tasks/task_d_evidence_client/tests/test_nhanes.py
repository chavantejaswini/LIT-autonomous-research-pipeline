"""NHANES source tests."""
from __future__ import annotations

from evidence_client.models import Failure, Success
from ._helpers import live_only


def test_nhanes_search_returns_success(client, vcr_cassette):
    r = client.nhanes.search(component="Demographics", cycle_year=2017)
    assert isinstance(r, Success)
    assert r.source == "nhanes"
    assert "html" in r.data


def test_nhanes_fetch_by_id_returns_success(client, vcr_cassette):
    r = client.nhanes.fetch_by_id(cycle="2017-2018", file_name="DEMO_J.htm")
    assert isinstance(r, Success)
    assert r.source == "nhanes"


@live_only()
def test_nhanes_live_search():
    from evidence_client import EvidenceClient

    c = EvidenceClient()
    r = c.nhanes.search(component="Demographics", cycle_year=2017)
    assert r.ok()
