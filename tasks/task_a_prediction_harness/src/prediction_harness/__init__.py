"""Prospective simulation harness with pre-registration.

A researcher registers a model prediction before the outcome is observed.
The harness makes backfilling impossible: predictions are append-only,
identified by the SHA-256 hash of their content, and outcomes whose
observation timestamp is not strictly after registration are rejected.
"""

from .api import Harness
from .errors import (
    HarnessError,
    ImmutablePredictionError,
    PredictionNotFoundError,
    TemporalOrderingError,
)
from .models import (
    CalibrationBin,
    CalibrationReport,
    PredictionId,
    PredictionRecord,
)

__all__ = [
    "Harness",
    "HarnessError",
    "ImmutablePredictionError",
    "PredictionNotFoundError",
    "TemporalOrderingError",
    "CalibrationBin",
    "CalibrationReport",
    "PredictionId",
    "PredictionRecord",
]
