"""Edge cases: input validation and persistence across DAO instances."""
from __future__ import annotations

from datetime import timedelta

import pytest

from prediction_harness import Harness
from prediction_harness.errors import InvalidPredictionShapeError
from prediction_harness.hashing import content_hash
from prediction_harness.sqlite_dao import SQLitePredictionStore


def test_prediction_without_probability_field_raises(harness: Harness) -> None:
    with pytest.raises(InvalidPredictionShapeError):
        harness.register_prediction("m", "d", {"score": 0.5})


def test_probability_out_of_range_raises(harness: Harness) -> None:
    with pytest.raises(InvalidPredictionShapeError):
        harness.register_prediction("m", "d", {"probability": 1.2})
    with pytest.raises(InvalidPredictionShapeError):
        harness.register_prediction("m", "d", {"probability": -0.1})


def test_outcome_label_must_be_zero_or_one(harness: Harness, clock) -> None:
    pid = harness.register_prediction("m", "d", {"probability": 0.5})
    clock.tick(60)
    with pytest.raises(InvalidPredictionShapeError):
        harness.record_outcome(pid, {"label": 2}, observed_at=clock())


def test_persistence_across_harness_instances(tmp_path, clock) -> None:
    """A file-backed store survives `Harness` instantiation churn."""
    url = f"sqlite:///{tmp_path / 'pred.db'}"
    h1 = Harness(store=SQLitePredictionStore(url=url), clock=clock)
    t0 = clock()
    pid = h1.register_prediction("m", "d", {"probability": 0.42})
    clock.tick(60)
    h1.record_outcome(pid, {"label": 1}, observed_at=clock())

    # New Harness + new Store, same DB file.
    h2 = Harness(store=SQLitePredictionStore(url=url), clock=clock)
    rec = h2.get_record(pid)
    assert rec.prediction == {"probability": 0.42}
    assert rec.outcome == {"label": 1}
    assert rec.registered_at == t0


def test_content_hash_matches_known_sha256() -> None:
    """The content hash must be the SHA-256 of the canonical envelope.

    This pins the hash format so changing it later is a breaking-change
    decision, not an accidental one.
    """
    import hashlib

    expected = hashlib.sha256(
        b'{"dataset_hash":"ds","model_id":"m","prediction":{"probability":0.5}}'
    ).hexdigest()
    assert content_hash("m", "ds", {"probability": 0.5}) == expected
