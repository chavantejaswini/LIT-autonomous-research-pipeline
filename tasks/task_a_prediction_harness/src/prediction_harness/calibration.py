"""Calibration math: 10-bin reliability curve, Brier score, ECE, plus
log-loss, accuracy, and MCE for the production-grade report.

Conventions enforced upstream by `api.Harness`:
  * Each prediction dict has a `probability` field, a float in [0, 1].
  * Each outcome dict has a `label` field, 0 or 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import CalibrationBin

NUM_BINS = 10


@dataclass(frozen=True)
class CalibrationMetrics:
    bins: list[CalibrationBin]
    brier: float
    ece: float
    log_loss: float
    accuracy_at_0_5: float
    max_calibration_error: float
    num_positives: int
    num_negatives: int


def compute_calibration(
    probs: np.ndarray, labels: np.ndarray
) -> CalibrationMetrics:
    """Return calibration metrics for the given paired arrays.

    Both arrays must have the same length and are assumed to be non-empty.
    The first bin spans [0.0, 0.1]; subsequent bins span (lower, upper].
    """
    if probs.shape != labels.shape:
        raise ValueError("probs and labels must have matching shapes")
    if probs.size == 0:
        raise ValueError("compute_calibration requires at least one sample")

    edges = np.linspace(0.0, 1.0, NUM_BINS + 1)
    bin_idx = np.digitize(probs, edges[1:-1], right=True)
    bin_idx = np.clip(bin_idx, 0, NUM_BINS - 1)

    bins: list[CalibrationBin] = []
    ece = 0.0
    mce = 0.0
    n = probs.size

    for b in range(NUM_BINS):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            bins.append(
                CalibrationBin(
                    bin_lower=float(edges[b]),
                    bin_upper=float(edges[b + 1]),
                    count=0,
                    mean_predicted=None,
                    empirical_frequency=None,
                )
            )
            continue
        mean_pred = float(probs[mask].mean())
        emp_freq = float(labels[mask].mean())
        gap = abs(mean_pred - emp_freq)
        bins.append(
            CalibrationBin(
                bin_lower=float(edges[b]),
                bin_upper=float(edges[b + 1]),
                count=count,
                mean_predicted=mean_pred,
                empirical_frequency=emp_freq,
            )
        )
        ece += (count / n) * gap
        if gap > mce:
            mce = gap

    brier = float(np.mean((probs - labels) ** 2))

    # Log-loss (binary cross-entropy). Clip probabilities so log doesn't blow up.
    eps = 1e-12
    p_clipped = np.clip(probs, eps, 1.0 - eps)
    log_loss = float(
        -np.mean(labels * np.log(p_clipped) + (1.0 - labels) * np.log(1.0 - p_clipped))
    )

    # Accuracy at threshold 0.5.
    accuracy_at_0_5 = float(((probs >= 0.5).astype(int) == labels.astype(int)).mean())
    num_positives = int(labels.sum())
    num_negatives = int(n - num_positives)

    return CalibrationMetrics(
        bins=bins,
        brier=brier,
        ece=float(ece),
        log_loss=log_loss,
        accuracy_at_0_5=accuracy_at_0_5,
        max_calibration_error=float(mce),
        num_positives=num_positives,
        num_negatives=num_negatives,
    )


def empty_curve() -> list[CalibrationBin]:
    """Return a curve with zero counts in every bin. Used when no outcomes exist."""
    edges = np.linspace(0.0, 1.0, NUM_BINS + 1)
    return [
        CalibrationBin(
            bin_lower=float(edges[b]),
            bin_upper=float(edges[b + 1]),
            count=0,
            mean_predicted=None,
            empirical_frequency=None,
        )
        for b in range(NUM_BINS)
    ]
