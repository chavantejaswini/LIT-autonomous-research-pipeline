"""Temporal-ordering invariants — backfilling must fail loudly."""
from __future__ import annotations

from datetime import timedelta, timezone

import pytest

from prediction_harness import (
    Harness,
    PredictionNotFoundError,
    TemporalOrderingError,
)


def test_outcome_before_registration_raises(harness: Harness, clock) -> None:
    t0 = clock()
    pid = harness.register_prediction("m", "d", {"probability": 0.5})
    with pytest.raises(TemporalOrderingError):
        harness.record_outcome(pid, {"label": 1}, observed_at=t0 - timedelta(seconds=1))


def test_outcome_equal_to_registration_raises(harness: Harness, clock) -> None:
    """`observed_at` must be STRICTLY after registration."""
    t0 = clock()
    pid = harness.register_prediction("m", "d", {"probability": 0.5})
    with pytest.raises(TemporalOrderingError):
        harness.record_outcome(pid, {"label": 1}, observed_at=t0)


def test_outcome_one_microsecond_after_is_valid(harness: Harness, clock) -> None:
    t0 = clock()
    pid = harness.register_prediction("m", "d", {"probability": 0.5})
    harness.record_outcome(
        pid, {"label": 1}, observed_at=t0 + timedelta(microseconds=1)
    )


def test_record_outcome_for_unknown_prediction_raises(harness: Harness, clock) -> None:
    with pytest.raises(PredictionNotFoundError):
        harness.record_outcome(
            "deadbeef" * 8, {"label": 1}, observed_at=clock() + timedelta(seconds=1)
        )


def test_timezone_aware_observed_at_is_normalized(harness: Harness, clock) -> None:
    """Aware datetimes should be compared after UTC normalization."""
    t0 = clock()
    pid = harness.register_prediction("m", "d", {"probability": 0.5})
    # t0 is naive UTC at 12:00. An aware datetime that resolves to t0 in UTC
    # must still trigger the temporal error.
    aware_equal = t0.replace(tzinfo=timezone.utc)
    with pytest.raises(TemporalOrderingError):
        harness.record_outcome(pid, {"label": 1}, observed_at=aware_equal)
