"""Pydantic models exposed at the harness boundary.

The DAO layer is free to use its own internal representation (ORM rows,
dataclasses, etc.); these models are what callers see.
"""
from __future__ import annotations

from datetime import datetime
from typing import NewType

from pydantic import BaseModel, ConfigDict, Field

# A PredictionId is the hex SHA-256 of the prediction's canonical envelope.
# Modeled as a NewType so we get type-checker support without runtime cost.
PredictionId = NewType("PredictionId", str)


class PredictionRecord(BaseModel):
    """A registered prediction, optionally with a recorded outcome."""

    model_config = ConfigDict(frozen=True)

    prediction_id: PredictionId
    model_id: str
    dataset_hash: str
    prediction: dict
    content_hash: str
    registered_at: datetime
    outcome: dict | None = None
    observed_at: datetime | None = None


class CalibrationBin(BaseModel):
    """One bin of a reliability diagram."""

    model_config = ConfigDict(frozen=True)

    bin_lower: float = Field(ge=0.0, le=1.0)
    bin_upper: float = Field(ge=0.0, le=1.0)
    count: int = Field(ge=0)
    mean_predicted: float | None = None
    empirical_frequency: float | None = None


class CalibrationReport(BaseModel):
    """Output of `calibration_report(...)`.

    Brief-required metrics: `num_registered`, `num_with_outcomes`,
    `calibration_curve` (10 bins), `brier_score`, `ece`.

    Production extensions: per-class realized counts, `log_loss`,
    `accuracy_at_0_5`, `max_calibration_error`. All optional and `None`
    when no outcomes have been observed in the window.
    """

    model_config = ConfigDict(frozen=True)

    model_id: str
    window_start: datetime
    window_end: datetime
    dataset_hash: str | None = None
    num_registered: int
    num_with_outcomes: int
    num_realized_positives: int = 0
    num_realized_negatives: int = 0
    calibration_curve: list[CalibrationBin]
    brier_score: float | None
    ece: float | None
    log_loss: float | None = None
    accuracy_at_0_5: float | None = None
    max_calibration_error: float | None = None
