"""Abstract storage layer.

A concrete `PredictionStore` is anything that can append predictions and
their outcomes while honoring the append-only invariant. The SQLite
implementation in `sqlite_dao.py` is the one shipped today; swapping in
Postgres means writing a new subclass that satisfies this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class StoredPrediction:
    """Internal representation of one row in the predictions table."""

    prediction_id: str
    model_id: str
    dataset_hash: str
    prediction_json: str
    content_hash: str
    registered_at: datetime


@dataclass(frozen=True)
class StoredOutcome:
    """Internal representation of one row in the outcomes table."""

    prediction_id: str
    outcome_json: str
    observed_at: datetime
    recorded_at: datetime


class PredictionStore(ABC):
    """Storage contract for the prediction harness.

    Implementations MUST enforce:
      * `add_prediction` is atomic — either the row is appended or nothing
        changes. Re-adding the same prediction_id raises.
      * `add_outcome` is atomic, and rejects a second outcome for the same
        prediction_id.
      * No method exposes an UPDATE on existing rows.
    """

    @abstractmethod
    def add_prediction(self, record: StoredPrediction) -> None: ...

    @abstractmethod
    def get_prediction(self, prediction_id: str) -> StoredPrediction | None: ...

    @abstractmethod
    def has_outcome(self, prediction_id: str) -> bool: ...

    @abstractmethod
    def add_outcome(self, outcome: StoredOutcome) -> None: ...

    @abstractmethod
    def get_outcome(self, prediction_id: str) -> StoredOutcome | None: ...

    @abstractmethod
    def list_predictions(
        self,
        model_id: str,
        window_start: datetime,
        window_end: datetime,
    ) -> list[StoredPrediction]: ...

    @abstractmethod
    def list_outcomes_for(
        self, prediction_ids: list[str]
    ) -> dict[str, StoredOutcome]: ...
