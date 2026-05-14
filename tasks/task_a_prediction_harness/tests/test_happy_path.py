"""Happy path: register, observe later, report."""
from __future__ import annotations

from datetime import timedelta

from prediction_harness import Harness


def test_register_returns_a_64_char_hex_id(harness: Harness) -> None:
    pid = harness.register_prediction(
        "model_v1", "ds_abc", {"probability": 0.7}
    )
    assert isinstance(pid, str)
    assert len(pid) == 64
    int(pid, 16)  # asserts hex


def test_round_trip_register_record_report(harness: Harness, clock) -> None:
    t0 = clock()
    pid = harness.register_prediction(
        "model_v1", "ds_abc", {"probability": 0.8}
    )
    clock.tick(60)  # advance the harness clock for the next call
    harness.record_outcome(pid, {"label": 1}, observed_at=t0 + timedelta(minutes=5))

    report = harness.calibration_report(
        "model_v1", (t0 - timedelta(hours=1), t0 + timedelta(hours=1))
    )
    assert report.num_registered == 1
    assert report.num_with_outcomes == 1
    assert report.brier_score is not None
    # p=0.8, y=1 → (0.8 - 1)^2 = 0.04
    assert abs(report.brier_score - 0.04) < 1e-9


def test_get_record_returns_prediction_and_outcome(harness: Harness, clock) -> None:
    t0 = clock()
    pid = harness.register_prediction("m", "d", {"probability": 0.3})
    clock.tick(60)
    harness.record_outcome(pid, {"label": 0}, observed_at=t0 + timedelta(minutes=2))
    rec = harness.get_record(pid)
    assert rec.prediction == {"probability": 0.3}
    assert rec.outcome == {"label": 0}
    assert rec.registered_at == t0
    assert rec.observed_at == t0 + timedelta(minutes=2)
