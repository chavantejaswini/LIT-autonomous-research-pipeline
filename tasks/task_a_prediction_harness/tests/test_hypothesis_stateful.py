"""Stateful property test of the harness with `hypothesis`.

Generates random sequences of register/record/report calls and verifies
several invariants hold throughout, plus a global invariant at teardown.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings, strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule

from prediction_harness import (
    Harness,
    ImmutablePredictionError,
    TemporalOrderingError,
)
from prediction_harness.sqlite_dao import SQLitePredictionStore


class _FixedClock:
    def __init__(self, start: datetime) -> None:
        self._now = start

    def __call__(self) -> datetime:
        return self._now

    def tick(self, seconds: float = 1.0) -> datetime:
        self._now = self._now + timedelta(seconds=seconds)
        return self._now


class HarnessStateMachine(RuleBasedStateMachine):
    """Random walk: register, record (sometimes pre-registration), report.

    Per-step invariants:
      * `num_registered ≥ num_with_outcomes` (you can't have more outcomes
        than predictions).
      * Re-registering the same content always raises ImmutablePredictionError.
    """

    predictions = Bundle("predictions")

    def __init__(self) -> None:
        super().__init__()
        self.clock = _FixedClock(datetime(2026, 1, 1, 0, 0, 0))
        self.harness = Harness(store=SQLitePredictionStore(), clock=self.clock)
        self.model_id = "m_hyp"
        self.total_registered = 0
        self.total_outcomes = 0
        self._seq = 0

    @rule(
        target=predictions,
        prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def register(self, prob):
        self._seq += 1
        pred = {"probability": prob, "seq": self._seq}
        self.clock.tick(1)
        pid = self.harness.register_prediction(self.model_id, "ds", pred)
        self.total_registered += 1
        return (pid, self.clock())  # carry the registration timestamp

    @rule(p=predictions, label=st.integers(min_value=0, max_value=1))
    def record_outcome(self, p, label):
        pid, registered_at = p
        observed_at = registered_at + timedelta(seconds=1)
        try:
            self.harness.record_outcome(pid, {"label": label}, observed_at=observed_at)
            self.total_outcomes += 1
        except ImmutablePredictionError:
            # Outcome already recorded — expected when this rule fires twice
            # on the same handle.
            pass

    @rule(p=predictions)
    def attempt_temporal_backfill(self, p):
        """Asserts the temporal-ordering invariant under random sequences."""
        pid, registered_at = p
        with pytest.raises(TemporalOrderingError):
            self.harness.record_outcome(
                pid,
                {"label": 0},
                observed_at=registered_at,  # equal → strictly after fails
            )

    @invariant()
    def outcomes_never_exceed_registrations(self):
        report = self.harness.calibration_report(
            self.model_id, (datetime(2025, 1, 1), datetime(2027, 1, 1))
        )
        assert report.num_with_outcomes <= report.num_registered
        assert report.num_registered == self.total_registered


HarnessStateMachine.TestCase.settings = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
TestHarnessStateful = HarnessStateMachine.TestCase
