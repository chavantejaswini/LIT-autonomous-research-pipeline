"""STRING-DB source tests."""
from __future__ import annotations

from evidence_client.models import Failure, Success
from ._helpers import live_only


def test_string_db_search_returns_success(client, vcr_cassette):
    r = client.string_db.search(["TP53"], species=9606)
    assert isinstance(r, Success)
    assert r.source == "string_db"


def test_string_db_fetch_by_id_returns_success(client, vcr_cassette):
    r = client.string_db.fetch_by_id("TP53", species=9606)
    assert isinstance(r, Success)
    assert r.source == "string_db"


@live_only()
def test_string_db_live_search():
    from evidence_client import EvidenceClient

    c = EvidenceClient()
    r = c.string_db.search(["TP53"], species=9606)
    assert r.ok()
