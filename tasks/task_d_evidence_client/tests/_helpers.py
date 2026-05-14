"""Test helpers exposed to all tests."""
from __future__ import annotations

import os

import pytest


def live_only():
    """Decorator: skip a live-integration test unless `EVIDENCE_LIVE=1`."""
    return pytest.mark.skipif(
        os.environ.get("EVIDENCE_LIVE") != "1",
        reason="set EVIDENCE_LIVE=1 to run live integration tests",
    )
