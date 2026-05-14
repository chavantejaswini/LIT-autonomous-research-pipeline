# LIT Test — Agentic System Developer Test Tasks

Five self-contained Python packages covering the technical test for the
Agentic System Developer role. Each task lives in its own directory under
`tasks/` and is independently installable and testable.

## Layout

```
tasks/
  task_a_prediction_harness/   Prospective simulation harness with pre-registration
  task_b_adversarial_screen/   Adversarial input detector at the institute perimeter
  task_c_priority_scheduler/   Compute-resource priority scheduler with pre-emption
  task_d_evidence_client/      Unified Result-typed biomedical evidence client
  task_e_cohort_generator/     Synthetic cohort generator from configurable distributions
```

## Quickstart

A single venv works for all five tasks:

```bash
python3 -m venv .venv
source .venv/bin/activate

for d in tasks/task_*; do
  (cd "$d" && pip install -e ".[dev]" --quiet && pytest -q)
done
```

Or per task:

```bash
cd tasks/task_a_prediction_harness
pip install -e ".[dev]"
pytest -v
cat README.md
```

## Test totals

| Task | Tests | Notes |
|------|-------|-------|
| **A** — Prediction harness | 27 passing | Append-only, content-addressed, calibration math vs known-truth fixtures. |
| **B** — Adversarial screen | 23 passing | Achieved: 0% FP, 100% recall, p99 ≈ 1ms on a 122-example corpus. |
| **C** — Priority scheduler | 11 passing | 50-workload stress: 0 priority inversions, every yield matched by resume, budget respected. |
| **D** — Evidence client | 22 passing + 5 live-gated | 10 VCR cassettes recorded against real PubMed / CTG / FAERS / NHANES / STRING-DB endpoints. |
| **E** — Cohort generator | 20 passing | KS-test marginals, ±0.05 correlations, ±2pp subtype proportions, byte-identical reproducibility — verified on three example disease configs. |
| **Total** | **103 passing + 5 gated** | All test suites run in under one second each. |

## Per-task summary

| Task | One-line summary |
|------|------------------|
| **A** | Append-only registration of model predictions with SHA-256 content hashing; immutability and temporal-ordering enforced at the SQLAlchemy DAO level; calibration report (Brier, ECE, 10-bin curve) over a time window. |
| **B** | Three-detector screening — rule-based directionality (JSON config), Mahalanobis anomaly on TF-IDF/SVD embeddings, logistic-regression classifier. Verdict aggregation lives entirely in a YAML config. |
| **C** | Strict-priority discrete-time scheduler with pre-emption + on-disk checkpoint/resume. Append-only audit log. 50-workload stress harness verifies all three required invariants. |
| **D** | Five biomedical APIs behind a single `Success | Failure` discriminated union. Configurable timeout, 429-Retry-After backoff with max 3 retries, structured logging, in-memory TTL cache. VCR-replay tests + gated live tests. |
| **E** | Gaussian-copula generator with analytical correlation calibration for non-linear marginals; stratified subtype assignment for tight proportion tolerance; three hand-authored example disease configs and a demo notebook. |

See each task's `README.md` for the full design and API surface.

## Dependencies

The combined venv installs:

- `pydantic`, `numpy`, `pandas`, `scipy`
- `sqlalchemy` (Task A)
- `scikit-learn`, `pyyaml`, `joblib` (Tasks B, C)
- `requests`, `vcrpy` (Task D)
- `jupyter`, `matplotlib` (Task E demo notebook — optional)
- `pytest` (all tasks)

No external services are required to run the test suites; Task D
ships pre-recorded cassettes so the run is fully offline.
