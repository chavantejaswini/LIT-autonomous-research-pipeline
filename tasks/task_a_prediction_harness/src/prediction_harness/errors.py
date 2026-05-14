"""Typed errors for the prediction harness."""


class HarnessError(Exception):
    """Base class for all harness-raised errors."""


class ImmutablePredictionError(HarnessError):
    """Raised when something tries to mutate an already-registered prediction.

    The harness is append-only. Re-registering the same content, recording
    an outcome twice for the same prediction, or any other mutation attempt
    surfaces as this error.
    """


class TemporalOrderingError(HarnessError):
    """Raised when an outcome's observation timestamp does not strictly
    follow the prediction's registration timestamp.

    The harness must reject backfills — outcomes observed at or before the
    moment a prediction was registered are not valid prospective tests.
    """


class PredictionNotFoundError(HarnessError):
    """Raised when an operation references a prediction that does not exist."""


class InvalidPredictionShapeError(HarnessError):
    """Raised when a prediction or outcome dict is missing required fields
    or has values outside the supported range."""
