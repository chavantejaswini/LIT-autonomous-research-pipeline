"""Top-level harness API.

`Harness` wraps a `PredictionStore` with the prospective-simulation
contract: append-only registration, strict temporal ordering between
registration and outcome, and calibration reporting over a time window.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from .calibration import compute_calibration, empty_curve
from .dao import PredictionStore, StoredOutcome, StoredPrediction
from .errors import (
    InvalidPredictionShapeError,
    PredictionNotFoundError,
    TemporalOrderingError,
)
from .hashing import canonical_json, content_hash
from .models import (
    CalibrationReport,
    PredictionId,
    PredictionRecord,
)
from .sqlite_dao import SQLitePredictionStore


def _utcnow() -> datetime:
    """Naive UTC `datetime` — SQLite stores TIMESTAMP as naive ISO strings."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _validate_prediction(prediction: dict) -> None:
    if "probability" not in prediction:
        raise InvalidPredictionShapeError(
            "prediction dict must include a 'probability' field in [0, 1]"
        )
    p = prediction["probability"]
    if not isinstance(p, (int, float)) or isinstance(p, bool):
        raise InvalidPredictionShapeError(
            "prediction['probability'] must be a number"
        )
    if not (0.0 <= float(p) <= 1.0):
        raise InvalidPredictionShapeError(
            f"prediction['probability'] must be in [0, 1], got {p}"
        )


def _validate_outcome(outcome: dict) -> None:
    if "label" not in outcome:
        raise InvalidPredictionShapeError(
            "outcome dict must include a 'label' field (0 or 1)"
        )
    label = outcome["label"]
    if label not in (0, 1, 0.0, 1.0, True, False):
        raise InvalidPredictionShapeError(
            f"outcome['label'] must be 0 or 1, got {label!r}"
        )


def _normalize_aware(dt: datetime) -> datetime:
    """Strip tz to a naive UTC datetime, matching the registered_at column."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class Harness:
    """Prospective simulation harness.

    Parameters
    ----------
    store : PredictionStore | None
        Storage backend. Defaults to an in-memory SQLite store, which is
        ideal for tests and ephemeral usage. Production callers should
        pass `SQLitePredictionStore(url="sqlite:///path.db")` or a
        Postgres-backed implementation.
    clock : Callable[[], datetime] | None
        Source of the registration timestamp. Override in tests to make
        timing assertions deterministic.
    """

    def __init__(
        self,
        store: PredictionStore | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store if store is not None else SQLitePredictionStore()
        self._clock = clock if clock is not None else _utcnow

    def register_prediction(
        self, model_id: str, dataset_hash: str, prediction: dict
    ) -> PredictionId:
        _validate_prediction(prediction)
        pid = content_hash(model_id, dataset_hash, prediction)
        record = StoredPrediction(
            prediction_id=pid,
            model_id=model_id,
            dataset_hash=dataset_hash,
            prediction_json=canonical_json(prediction),
            content_hash=pid,
            registered_at=self._clock(),
        )
        # add_prediction raises ImmutablePredictionError on collision.
        self._store.add_prediction(record)
        return PredictionId(pid)

    def record_outcome(
        self,
        prediction_id: PredictionId,
        outcome: dict,
        observed_at: datetime,
    ) -> None:
        _validate_outcome(outcome)
        observed_naive = _normalize_aware(observed_at)
        existing = self._store.get_prediction(prediction_id)
        if existing is None:
            raise PredictionNotFoundError(
                f"No prediction registered with id {prediction_id}"
            )
        if observed_naive <= existing.registered_at:
            raise TemporalOrderingError(
                f"Outcome observed_at ({observed_naive.isoformat()}) must be "
                f"strictly after registered_at ({existing.registered_at.isoformat()}). "
                "Backfilling is not permitted."
            )
        stored_outcome = StoredOutcome(
            prediction_id=prediction_id,
            outcome_json=canonical_json(outcome),
            observed_at=observed_naive,
            recorded_at=self._clock(),
        )
        # add_outcome raises ImmutablePredictionError on duplicate.
        self._store.add_outcome(stored_outcome)

    def calibration_report(
        self,
        model_id: str,
        time_window: tuple[datetime, datetime],
        dataset_hash: str | None = None,
    ) -> CalibrationReport:
        """Return a calibration report for `model_id` over the time window.

        If `dataset_hash` is provided, the report is restricted to
        predictions whose `dataset_hash` matches exactly. This lets a
        researcher compare a model's calibration across different
        evaluation datasets without re-querying the DAO.
        """
        start, end = (_normalize_aware(time_window[0]), _normalize_aware(time_window[1]))
        if end < start:
            raise ValueError("time_window end must be >= start")

        predictions = self._store.list_predictions(model_id, start, end)
        if dataset_hash is not None:
            predictions = [p for p in predictions if p.dataset_hash == dataset_hash]
        num_registered = len(predictions)
        if num_registered == 0:
            return CalibrationReport(
                model_id=model_id,
                window_start=start,
                window_end=end,
                dataset_hash=dataset_hash,
                num_registered=0,
                num_with_outcomes=0,
                calibration_curve=empty_curve(),
                brier_score=None,
                ece=None,
            )

        outcomes = self._store.list_outcomes_for(
            [p.prediction_id for p in predictions]
        )
        paired_probs: list[float] = []
        paired_labels: list[float] = []
        for p in predictions:
            o = outcomes.get(p.prediction_id)
            if o is None:
                continue
            prob = float(json.loads(p.prediction_json)["probability"])
            label = float(json.loads(o.outcome_json)["label"])
            paired_probs.append(prob)
            paired_labels.append(label)

        if not paired_probs:
            return CalibrationReport(
                model_id=model_id,
                window_start=start,
                window_end=end,
                dataset_hash=dataset_hash,
                num_registered=num_registered,
                num_with_outcomes=0,
                calibration_curve=empty_curve(),
                brier_score=None,
                ece=None,
            )

        metrics = compute_calibration(
            np.asarray(paired_probs, dtype=float),
            np.asarray(paired_labels, dtype=float),
        )
        return CalibrationReport(
            model_id=model_id,
            window_start=start,
            window_end=end,
            dataset_hash=dataset_hash,
            num_registered=num_registered,
            num_with_outcomes=len(paired_probs),
            num_realized_positives=metrics.num_positives,
            num_realized_negatives=metrics.num_negatives,
            calibration_curve=metrics.bins,
            brier_score=metrics.brier,
            ece=metrics.ece,
            log_loss=metrics.log_loss,
            accuracy_at_0_5=metrics.accuracy_at_0_5,
            max_calibration_error=metrics.max_calibration_error,
        )

    def get_record(self, prediction_id: PredictionId) -> PredictionRecord:
        """Convenience accessor — returns the prediction + its outcome (if any)."""
        p = self._store.get_prediction(prediction_id)
        if p is None:
            raise PredictionNotFoundError(prediction_id)
        o = self._store.get_outcome(prediction_id)
        return PredictionRecord(
            prediction_id=PredictionId(p.prediction_id),
            model_id=p.model_id,
            dataset_hash=p.dataset_hash,
            prediction=json.loads(p.prediction_json),
            content_hash=p.content_hash,
            registered_at=p.registered_at,
            outcome=json.loads(o.outcome_json) if o else None,
            observed_at=o.observed_at if o else None,
        )
