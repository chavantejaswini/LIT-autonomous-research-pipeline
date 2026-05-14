"""Immutability invariants — every mutation attempt must raise."""
from __future__ import annotations

from datetime import timedelta

import pytest

from prediction_harness import (
    Harness,
    ImmutablePredictionError,
)


def test_double_register_same_content_raises(harness: Harness) -> None:
    harness.register_prediction("model_a", "ds1", {"probability": 0.6})
    with pytest.raises(ImmutablePredictionError):
        harness.register_prediction("model_a", "ds1", {"probability": 0.6})


def test_double_register_different_dict_order_still_raises(harness: Harness) -> None:
    """Dict key order must not affect identity — canonical JSON is sorted."""
    harness.register_prediction(
        "model_a", "ds1", {"probability": 0.6, "meta": {"a": 1, "b": 2}}
    )
    with pytest.raises(ImmutablePredictionError):
        harness.register_prediction(
            "model_a", "ds1", {"meta": {"b": 2, "a": 1}, "probability": 0.6}
        )


def test_record_outcome_twice_raises(harness: Harness, clock) -> None:
    t0 = clock()
    pid = harness.register_prediction("m", "d", {"probability": 0.5})
    clock.tick(60)
    harness.record_outcome(pid, {"label": 1}, observed_at=t0 + timedelta(minutes=1))
    with pytest.raises(ImmutablePredictionError):
        harness.record_outcome(pid, {"label": 0}, observed_at=t0 + timedelta(minutes=2))


def test_different_prediction_value_yields_different_id(harness: Harness) -> None:
    pid1 = harness.register_prediction("m", "d", {"probability": 0.3})
    pid2 = harness.register_prediction("m", "d", {"probability": 0.4})
    assert pid1 != pid2


def test_id_is_deterministic_across_harness_instances(clock) -> None:
    from prediction_harness import Harness as H
    from prediction_harness.sqlite_dao import SQLitePredictionStore

    h1 = H(store=SQLitePredictionStore(), clock=clock)
    h2 = H(store=SQLitePredictionStore(), clock=clock)
    pid1 = h1.register_prediction("m", "d", {"probability": 0.42})
    pid2 = h2.register_prediction("m", "d", {"probability": 0.42})
    assert pid1 == pid2  # PredictionId derives only from content
