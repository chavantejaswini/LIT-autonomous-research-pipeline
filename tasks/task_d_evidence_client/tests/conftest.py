"""Shared test fixtures + VCR config."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import vcr

from evidence_client import EvidenceClient, TTLCache

CASSETTE_DIR = Path(__file__).parent / "cassettes"
CASSETTE_DIR.mkdir(exist_ok=True)


def make_vcr() -> vcr.VCR:
    """Single source of truth for VCR settings.

    `record_mode` is controlled by env: set `VCR_RECORD=1` to re-record
    cassettes against the live APIs. Otherwise we replay from disk only
    (`none` mode — any new request fails the test loudly).
    """
    record_mode = "new_episodes" if os.environ.get("VCR_RECORD") == "1" else "none"
    return vcr.VCR(
        cassette_library_dir=str(CASSETTE_DIR),
        record_mode=record_mode,
        match_on=("method", "scheme", "host", "path", "query"),
        filter_query_parameters=("api_key",),
        filter_headers=("authorization", "cookie"),
    )


@pytest.fixture
def vcr_cassette(request):
    """A VCR context whose cassette name is derived from the test function."""
    name = request.node.name + ".yaml"
    with make_vcr().use_cassette(name):
        yield


@pytest.fixture
def client():
    # Fresh cache each test so cached responses from a previous test don't
    # bypass the cassette.
    return EvidenceClient(cache=TTLCache(ttl_seconds=60.0))


def live_only():
    """Decorator: skip a live-integration test unless `EVIDENCE_LIVE=1`."""
    return pytest.mark.skipif(
        os.environ.get("EVIDENCE_LIVE") != "1",
        reason="set EVIDENCE_LIVE=1 to run live integration tests",
    )
