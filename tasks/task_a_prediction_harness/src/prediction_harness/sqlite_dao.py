"""SQLAlchemy/SQLite implementation of the prediction store.

The schema relies on PRIMARY KEY and UNIQUE constraints to enforce
immutability at the database level. There is no UPDATE statement issued
anywhere in this module — all writes are INSERTs that the database
rejects on collision.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .dao import PredictionStore, StoredOutcome, StoredPrediction
from .errors import ImmutablePredictionError


class _Base(DeclarativeBase):
    pass


class _PredictionRow(_Base):
    __tablename__ = "predictions"
    prediction_id = Column(String(64), primary_key=True)
    model_id = Column(String(256), nullable=False, index=True)
    dataset_hash = Column(String(256), nullable=False)
    prediction_json = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, unique=True)
    registered_at = Column(DateTime, nullable=False, index=True)


class _OutcomeRow(_Base):
    __tablename__ = "outcomes"
    # prediction_id is BOTH primary key AND foreign-key-like reference.
    # Using it as PK makes "one outcome per prediction" structural.
    prediction_id = Column(String(64), primary_key=True)
    outcome_json = Column(Text, nullable=False)
    observed_at = Column(DateTime, nullable=False)
    recorded_at = Column(DateTime, nullable=False)


class SQLitePredictionStore(PredictionStore):
    """SQLite-backed store. Pass `url=':memory:'` (the default) for tests."""

    def __init__(self, url: str = "sqlite:///:memory:") -> None:
        # `check_same_thread=False` lets the same in-memory DB be used from
        # different threads if the caller wants — pytest sometimes does.
        connect_args: dict = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        self._engine = create_engine(url, connect_args=connect_args, future=True)
        _Base.metadata.create_all(self._engine)
        self._SessionLocal = sessionmaker(self._engine, expire_on_commit=False)

    def _session(self) -> Session:
        return self._SessionLocal()

    def add_prediction(self, record: StoredPrediction) -> None:
        row = _PredictionRow(
            prediction_id=record.prediction_id,
            model_id=record.model_id,
            dataset_hash=record.dataset_hash,
            prediction_json=record.prediction_json,
            content_hash=record.content_hash,
            registered_at=record.registered_at,
        )
        with self._session() as s:
            s.add(row)
            try:
                s.commit()
            except IntegrityError as exc:
                s.rollback()
                raise ImmutablePredictionError(
                    f"Prediction {record.prediction_id} is already registered; "
                    "predictions are append-only."
                ) from exc

    def get_prediction(self, prediction_id: str) -> StoredPrediction | None:
        with self._session() as s:
            row = s.get(_PredictionRow, prediction_id)
            if row is None:
                return None
            return _to_stored_prediction(row)

    def has_outcome(self, prediction_id: str) -> bool:
        with self._session() as s:
            return s.get(_OutcomeRow, prediction_id) is not None

    def add_outcome(self, outcome: StoredOutcome) -> None:
        row = _OutcomeRow(
            prediction_id=outcome.prediction_id,
            outcome_json=outcome.outcome_json,
            observed_at=outcome.observed_at,
            recorded_at=outcome.recorded_at,
        )
        with self._session() as s:
            s.add(row)
            try:
                s.commit()
            except IntegrityError as exc:
                s.rollback()
                raise ImmutablePredictionError(
                    f"Outcome for prediction {outcome.prediction_id} is already "
                    "recorded; outcomes are append-only."
                ) from exc

    def get_outcome(self, prediction_id: str) -> StoredOutcome | None:
        with self._session() as s:
            row = s.get(_OutcomeRow, prediction_id)
            if row is None:
                return None
            return _to_stored_outcome(row)

    def list_predictions(
        self,
        model_id: str,
        window_start: datetime,
        window_end: datetime,
    ) -> list[StoredPrediction]:
        stmt = (
            select(_PredictionRow)
            .where(_PredictionRow.model_id == model_id)
            .where(_PredictionRow.registered_at >= window_start)
            .where(_PredictionRow.registered_at <= window_end)
            .order_by(_PredictionRow.registered_at.asc())
        )
        with self._session() as s:
            return [_to_stored_prediction(r) for r in s.scalars(stmt).all()]

    def list_outcomes_for(
        self, prediction_ids: list[str]
    ) -> dict[str, StoredOutcome]:
        if not prediction_ids:
            return {}
        stmt = select(_OutcomeRow).where(
            _OutcomeRow.prediction_id.in_(prediction_ids)
        )
        with self._session() as s:
            return {r.prediction_id: _to_stored_outcome(r) for r in s.scalars(stmt).all()}


def _to_stored_prediction(row: _PredictionRow) -> StoredPrediction:
    return StoredPrediction(
        prediction_id=row.prediction_id,
        model_id=row.model_id,
        dataset_hash=row.dataset_hash,
        prediction_json=row.prediction_json,
        content_hash=row.content_hash,
        registered_at=row.registered_at,
    )


def _to_stored_outcome(row: _OutcomeRow) -> StoredOutcome:
    return StoredOutcome(
        prediction_id=row.prediction_id,
        outcome_json=row.outcome_json,
        observed_at=row.observed_at,
        recorded_at=row.recorded_at,
    )
