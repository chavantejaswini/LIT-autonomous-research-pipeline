"""Shared fixtures for Task C tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from priority_scheduler import Scheduler

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "priorities.yaml"


@pytest.fixture
def scheduler(tmp_path):
    return Scheduler(
        config_path=CONFIG,
        budget=100_000,
        slots=2,
        audit_log_path=tmp_path / "audit.jsonl",
        checkpoint_dir=tmp_path / "ckpts",
    )
