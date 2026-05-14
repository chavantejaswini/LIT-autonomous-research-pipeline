"""Shared fixtures for Task A tests."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable

import pytest

from prediction_harness import Harness
from prediction_harness.sqlite_dao import SQLitePredictionStore


class FixedClock:
    """Deterministic clock for tests. `tick()` advances by 1 second by default."""

    def __init__(self, start: datetime) -> None:
        self._now = start

    def __call__(self) -> datetime:
        return self._now

    def tick(self, seconds: float = 1.0) -> datetime:
        self._now = self._now + timedelta(seconds=seconds)
        return self._now


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(datetime(2026, 1, 1, 12, 0, 0))


@pytest.fixture
def harness(clock: FixedClock) -> Harness:
    return Harness(store=SQLitePredictionStore(), clock=clock)


@pytest.fixture
def file_harness(tmp_path, clock: FixedClock) -> Harness:
    """Harness backed by a file-on-disk SQLite DB, to exercise the real DAO path."""
    url = f"sqlite:///{tmp_path / 'predictions.db'}"
    return Harness(store=SQLitePredictionStore(url=url), clock=clock)
