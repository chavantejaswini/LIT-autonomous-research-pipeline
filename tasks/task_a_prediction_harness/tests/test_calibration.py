"""Calibration math correctness against known-truth fixtures."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from prediction_harness import Harness
from prediction_harness.calibration import compute_calibration


def test_brier_known_truth_perfect_predictions() -> None:
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    labels = np.array([1, 0, 1, 0])
    m = compute_calibration(probs, labels)
    assert m.brier == 0.0
    assert m.ece == 0.0
    assert m.accuracy_at_0_5 == 1.0
    assert m.log_loss < 1e-6


def test_brier_known_truth_worst_case() -> None:
    """All predictions wrong with full confidence ⇒ Brier = 1."""
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    labels = np.array([0, 1, 0, 1])
    m = compute_calibration(probs, labels)
    assert m.brier == 1.0
    assert m.ece == 1.0
    assert m.accuracy_at_0_5 == 0.0


def test_brier_known_truth_uniform_uncertainty() -> None:
    """p=0.5 for everything, mixed labels ⇒ Brier = 0.25."""
    probs = np.full(10, 0.5)
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    m = compute_calibration(probs, labels)
    assert abs(m.brier - 0.25) < 1e-9
    # Mean predicted in the 0.4–0.5 bin is 0.5; empirical is 0.5; ECE = 0.
    assert abs(m.ece) < 1e-9


def test_calibration_curve_has_ten_bins() -> None:
    probs = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    labels = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    m = compute_calibration(probs, labels)
    assert len(m.bins) == 10
    # Each input lives in its own bin → count == 1 for each.
    assert sum(b.count for b in m.bins) == 10
    for b in m.bins:
        assert b.count == 1


def test_ece_matches_manual_calculation() -> None:
    """Two bins populated; verify ECE = (n_b/N) * |mean_pred - emp_freq| summed."""
    probs = np.array([0.2, 0.2, 0.8, 0.8])  # two in bin [.1,.2], two in bin [.7,.8]
    labels = np.array([0, 1, 1, 1])
    m = compute_calibration(probs, labels)
    # bin1: mean_pred=0.2, emp=0.5 → contribution = 0.5 * 0.3 = 0.15
    # bin2: mean_pred=0.8, emp=1.0 → contribution = 0.5 * 0.2 = 0.10
    assert abs(m.ece - 0.25) < 1e-9
    # MCE = max bin gap = 0.3
    assert abs(m.max_calibration_error - 0.3) < 1e-9


def test_report_through_harness_against_truth(harness: Harness, clock) -> None:
    """End-to-end: feed paired (prob, label) through the API and check Brier."""
    t0 = clock()
    samples = [(0.1, 0), (0.3, 0), (0.5, 1), (0.7, 1), (0.9, 1)]
    for i, (p, y) in enumerate(samples):
        pid = harness.register_prediction(
            "m", "ds", {"probability": p, "i": i}
        )
        clock.tick(60)
        harness.record_outcome(pid, {"label": y}, observed_at=clock())

    report = harness.calibration_report(
        "m", (t0 - timedelta(hours=1), clock() + timedelta(hours=1))
    )
    assert report.num_registered == 5
    assert report.num_with_outcomes == 5
    # Manual Brier: ((0.1)^2 + (0.3)^2 + (0.5-1)^2 + (0.7-1)^2 + (0.9-1)^2)/5
    expected = (0.01 + 0.09 + 0.25 + 0.09 + 0.01) / 5
    assert abs(report.brier_score - expected) < 1e-9


def test_empty_window_returns_zeros_not_an_error(harness: Harness) -> None:
    """A model with no predictions yields a structurally valid empty report."""
    report = harness.calibration_report(
        "ghost_model", (datetime(2026, 1, 1), datetime(2027, 1, 1))
    )
    assert report.num_registered == 0
    assert report.num_with_outcomes == 0
    assert report.brier_score is None
    assert report.ece is None
    assert len(report.calibration_curve) == 10


def test_predictions_without_outcomes_not_counted_in_brier(harness: Harness, clock) -> None:
    t0 = clock()
    # Two predictions: one gets an outcome, one does not.
    pid1 = harness.register_prediction("m", "ds", {"probability": 0.5, "i": 1})
    clock.tick(10)
    harness.record_outcome(pid1, {"label": 1}, observed_at=clock())
    harness.register_prediction("m", "ds", {"probability": 0.5, "i": 2})

    report = harness.calibration_report(
        "m", (t0 - timedelta(hours=1), clock() + timedelta(hours=1))
    )
    assert report.num_registered == 2
    assert report.num_with_outcomes == 1
    # Only the realized pair contributes: (0.5 - 1)^2 = 0.25
    assert abs(report.brier_score - 0.25) < 1e-9


def test_window_filters_by_registered_at(harness: Harness, clock) -> None:
    t0 = clock()
    harness.register_prediction("m", "ds", {"probability": 0.5, "i": 1})
    clock.tick(3600)  # +1h
    harness.register_prediction("m", "ds", {"probability": 0.5, "i": 2})

    early_report = harness.calibration_report(
        "m", (t0 - timedelta(minutes=5), t0 + timedelta(minutes=30))
    )
    assert early_report.num_registered == 1

    late_report = harness.calibration_report(
        "m", (t0 + timedelta(minutes=30), t0 + timedelta(hours=2))
    )
    assert late_report.num_registered == 1
