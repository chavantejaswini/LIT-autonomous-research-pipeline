# Task A — Prospective Simulation Harness with Pre-Registration

A Python module that lets a researcher register a model prediction *before*
the outcome is observed, then later record the observed outcome and produce
a calibration report. The harness is **append-only**, **content-addressed**,
and **temporally strict** — backfilling is impossible by construction.

**Production-grade:** thread-safe concurrent registration, extended
calibration metrics (log-loss, accuracy, MCE), optional `dataset_hash`
filtering on reports, a `prediction-harness` CLI, and `hypothesis`-driven
stateful property tests.

## Install & run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest -v

# CLI
prediction-harness --db pred.db register \
    --model-id cardio_v3 --dataset-hash sha256:abc \
    --prediction-json '{"probability": 0.74}'
prediction-harness --db pred.db record \
    --prediction-id <pid> --outcome-json '{"label": 1}' \
    --observed-at 2026-06-01T12:00:00
prediction-harness --db pred.db report \
    --model-id cardio_v3 --start 2026-01-01T00:00:00 --end 2026-12-31T23:59:59
```

## API at a glance

```python
from prediction_harness import Harness

h = Harness()  # in-memory SQLite by default

pid = h.register_prediction(
    model_id="cardio_risk_v3",
    dataset_hash="sha256:abc123...",
    prediction={"probability": 0.74, "subject": "S-001"},
)

h.record_outcome(
    pid, outcome={"label": 1}, observed_at=datetime(2026, 6, 1, 12, 0),
)

report = h.calibration_report(
    "cardio_risk_v3",
    time_window=(datetime(2026, 1, 1), datetime(2026, 12, 31)),
    dataset_hash="sha256:abc123...",  # optional filter
)
print(report.brier_score, report.ece, report.log_loss,
      report.accuracy_at_0_5, report.max_calibration_error)
```

### Required prediction / outcome shape

Every `prediction` dict must include a `"probability"` field in `[0, 1]`.
Every `outcome` dict must include a `"label"` of `0` or `1`. Both dicts
can carry any additional fields and those fields participate in the content hash.

## Design

### Append-only with SHA-256 content hash

`PredictionId` is the hex SHA-256 of the canonical JSON envelope:

```
sha256({"model_id": ..., "dataset_hash": ..., "prediction": ...})
```

with sorted keys and no whitespace. Same `(model_id, dataset_hash, prediction)`
always produces the same ID regardless of dict ordering.

The database enforces immutability with `PRIMARY KEY` and `UNIQUE`
constraints; the DAO never issues an `UPDATE` statement. Any attempt to
re-register the same content, or to record a second outcome for a
prediction, surfaces as `ImmutablePredictionError`.

### Temporal ordering

`record_outcome` rejects any `observed_at` that is not **strictly after**
the prediction's `registered_at`. A microsecond suffices; equality fails.
Timezone-aware datetimes are normalized to UTC before comparison.

### Storage

The DAO interface (`prediction_harness.dao.PredictionStore`) is abstract.
The shipped implementation is `SQLitePredictionStore` (SQLAlchemy + SQLite).
A Postgres backend is a drop-in subclass.

**Concurrent-write safety:** the SQLite `PRIMARY KEY` + `UNIQUE` constraints
guarantee that if 16 threads race to register the same content, exactly one
wins and 15 see `ImmutablePredictionError`. Verified in `test_concurrency.py`.

### Calibration report

`CalibrationReport` returns:

| Field | Description |
|-------|-------------|
| `num_registered` | Predictions for this `model_id` in the window. |
| `num_with_outcomes` | Subset that have a recorded outcome. |
| `num_realized_positives` | Among realized outcomes, how many had `label=1`. |
| `num_realized_negatives` | Among realized outcomes, how many had `label=0`. |
| `dataset_hash` | If filtered, the dataset_hash applied; otherwise `None`. |
| `calibration_curve` | 10 bins over `[0, 1]`, each with `count`, `mean_predicted`, `empirical_frequency`. |
| `brier_score` | `mean((p − y)²)` over realized pairs. `None` when no outcomes. |
| `ece` | `Σ (n_b / N) · |mean_pred_b − emp_freq_b|`. `None` when no outcomes. |
| `log_loss` | Binary cross-entropy with `eps`-clipped probs. |
| `accuracy_at_0_5` | Fraction where `(prob ≥ 0.5) == label`. |
| `max_calibration_error` | Largest per-bin `|mean_pred − emp_freq|`. |

### CLI

`prediction-harness` exposes three subcommands (`register`, `record`,
`report`) that mirror the Python API. All input is JSON; all output is
JSON. Non-zero exit codes plus a structured error body surface
`HarnessError` subclasses (see `cli.py`).

## Tests

**35 tests** across the suite:

- `test_happy_path.py` (3) — round-trip register → record → report.
- `test_immutability.py` (5) — double-register, dict-order-invariance, double-outcome, deterministic IDs.
- `test_temporal_ordering.py` (5) — strict-after invariant, equality fails, microsecond-after succeeds, timezone normalization, unknown prediction id.
- `test_calibration.py` (9) — known-truth fixtures for Brier / ECE / log-loss / accuracy / MCE, 10-bin curve, manual-calculation cross-check, empty-window handling, predictions-without-outcomes excluded, window filtering.
- `test_validation_and_storage.py` (5) — input validation, file-backed persistence, content-hash format pinned to a known SHA-256.
- `test_extended_metrics.py` (3) — production metrics populated; `dataset_hash` filter narrows the report; log-loss penalises overconfident wrong predictions more than Brier.
- `test_cli.py` (2) — end-to-end CLI register → record → report; `--dataset-hash` filter.
- `test_concurrency.py` (2) — 16 threads racing same content yield 1 winner + 15 `ImmutablePredictionError`; 16 distinct registrations all succeed.
- `test_hypothesis_stateful.py` (1) — random sequences of register / record / report verified against an in-memory shadow of the state machine.

## Dependencies

`pydantic`, `sqlalchemy`, `numpy` (plus `pytest` + `hypothesis` for tests). No others.
