"""Adversarial input screening at the institute perimeter.

Three independent detectors examine each incoming research directive:

  1. A rule-based directionality check (JSON config).
  2. A Mahalanobis-distance anomaly detector fitted on benign embeddings.
  3. An ML classifier trained on a synthetic adversarial corpus.

Their subscores are combined by a YAML-configured aggregator into an
ALLOW / REVIEW / BLOCK verdict with a structured explanation.
"""

from .models import (
    DetectorSubscore,
    InputSource,
    InstituteInput,
    ScreenResult,
    Verdict,
)
from .screener import Screener

__all__ = [
    "DetectorSubscore",
    "InputSource",
    "InstituteInput",
    "ScreenResult",
    "Screener",
    "Verdict",
]
