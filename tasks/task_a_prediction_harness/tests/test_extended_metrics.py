"""Extended report fields: log_loss, accuracy, MCE, realized class counts,
and the optional `dataset_hash` filter on `calibration_report`."""
from __future__ import annotations

from datetime import timedelta

from prediction_harness import Harness


def _record_pairs(harness, model_id, dataset_hash, samples, clock):
    """Helper: register + record a list of (prob, label, i) tuples."""
    for p, y, i in samples:
        pid = harness.register_prediction(
            model_id, dataset_hash, {"probability": p, "i": i}
        )
        clock.tick(60)
        harness.record_outcome(pid, {"label": y}, observed_at=clock())


def test_extended_metrics_populated(harness, clock):
    t0 = clock()
    _record_pairs(
        harness, "m", "ds",
        [(0.1, 0, 0), (0.3, 0, 1), (0.5, 1, 2), (0.7, 1, 3), (0.9, 1, 4)],
        clock,
    )
    report = harness.calibration_report(
        "m", (t0 - timedelta(hours=1), clock() + timedelta(hours=1))
    )
    assert report.num_realized_positives == 3
    assert report.num_realized_negatives == 2
    assert report.log_loss is not None and report.log_loss > 0
    assert report.accuracy_at_0_5 is not None
    # threshold 0.5 → predicted [0,0,1,1,1], actual [0,0,1,1,1] → 100% accuracy
    assert report.accuracy_at_0_5 == 1.0
    assert report.max_calibration_error is not None
    assert 0 <= report.max_calibration_error <= 1


def test_dataset_hash_filter(harness, clock):
    t0 = clock()
    # Two datasets, both for the same model. Different prediction values.
    _record_pairs(harness, "m", "ds_A", [(0.9, 1, 0), (0.9, 1, 1)], clock)
    _record_pairs(harness, "m", "ds_B", [(0.1, 0, 0), (0.1, 0, 1)], clock)

    window = (t0 - timedelta(hours=1), clock() + timedelta(hours=1))
    report_a = harness.calibration_report("m", window, dataset_hash="ds_A")
    report_b = harness.calibration_report("m", window, dataset_hash="ds_B")
    report_all = harness.calibration_report("m", window)

    assert report_a.num_registered == 2
    assert report_b.num_registered == 2
    assert report_all.num_registered == 4
    assert report_a.dataset_hash == "ds_A"
    assert report_b.dataset_hash == "ds_B"
    assert report_all.dataset_hash is None
    # Per-dataset reports have very different Brier scores.
    assert abs(report_a.brier_score - (0.1 ** 2)) < 1e-9
    assert abs(report_b.brier_score - (0.1 ** 2)) < 1e-9


def test_log_loss_higher_for_overconfident_wrong_predictions(harness, clock):
    """A model that says 0.99 and is wrong has very high log-loss."""
    t0 = clock()
    _record_pairs(harness, "m", "ds", [(0.99, 0, 0), (0.99, 0, 1)], clock)
    report = harness.calibration_report(
        "m", (t0 - timedelta(hours=1), clock() + timedelta(hours=1))
    )
    # log_loss = -log(1 - 0.99) ≈ 4.6 for each → mean ≈ 4.6
    assert report.log_loss > 4.0
