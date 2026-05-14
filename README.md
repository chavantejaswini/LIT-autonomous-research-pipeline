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
| **A** — Prediction harness | 35 passing | Append-only, content-addressed, calibration math vs known-truth fixtures, 16-thread concurrency stress, Hypothesis stateful invariants. |
| **B** — Adversarial screen | 36 passing | Under strictest tier: 3.28% FP, 100% recall, p99 ≈ 0.65 ms on a 122-example corpus. 5-fold stratified CV at training. |
| **C** — Priority scheduler | 23 passing | 100-seed stress sweep: 5000/5000 workloads completed, 108 pre-emptions, 0 priority inversions. Hypothesis stateful random-walk. |
| **D** — Evidence client | 47 passing + 5 live-gated | 10 VCR cassettes recorded against real PubMed / CTG / FAERS / NHANES / STRING-DB endpoints. Per-source circuit breakers and metrics. |
| **E** — Cohort generator | 50 passing | KS-test marginals, ±0.05 correlations, ±2pp subtype proportions, byte-identical reproducibility, MCAR/MAR missingness, Hypothesis-fuzzed configs. |
| **Total** | **191 passing + 5 gated** | All test suites run in under seven seconds each. |

## Per-task summary

| Task | One-line summary |
|------|------------------|
| **A** | Append-only registration of model predictions with SHA-256 content hashing; immutability and temporal-ordering enforced at the SQLAlchemy DAO level; calibration report (Brier, ECE, log-loss, accuracy, MCE) over a time window. |
| **B** | Three-detector screening — rule-based directionality (JSON config), Mahalanobis anomaly on TF-IDF/SVD embeddings, logistic-regression classifier. Verdict aggregation and per-source thresholds live entirely in a YAML config. |
| **C** | Strict-priority discrete-time scheduler with pre-emption + on-disk checkpoint/resume. Append-only audit log. Pluggable runtime (sync default, threaded production). 100-seed stress sweep verifies all three required invariants. |
| **D** | Five biomedical APIs behind a single `Success`/`Failure` discriminated union. Configurable timeout, 429+5xx Retry-After backoff, per-source circuit breakers, thread-safe TTL cache, in-memory metrics, pydantic response schemas. |
| **E** | Gaussian-copula generator with analytical Pearson calibration for non-linear marginals; stratified subtype assignment for tight proportion tolerance; biomarker profile shifts, MCAR/MAR missingness, streaming generation, provenance manifest. |

See each task's `README.md` for the full design and API surface.

---

## Task A — Prospective Simulation Harness

**Problem.** Researchers register model predictions *before* outcomes are
observed. The system has to make backfilling impossible, surface a
calibration report after enough outcomes land, and survive concurrent
writers from many notebooks at once.

### Smart approaches

- **Content-addressed identifiers.** `PredictionId = SHA-256 hex of
  canonical_json({model_id, dataset_hash, prediction})`. The id becomes a
  pure function of input content, which means (a) the same prediction
  registered twice maps to the same row and is deterministically rejected,
  and (b) any tampering with the stored prediction breaks the id-content
  relationship and is detectable on read.
- **Append-only enforced at the DB layer**, not the application layer. The
  SQLAlchemy DAO never issues an `UPDATE` against the predictions table;
  immutability rides on `PRIMARY KEY` plus `UNIQUE` constraints. Race
  conditions are won by SQLite's transactional commit, not by an
  application-side lock that could be bypassed.
- **Strict-after temporal check.** `observed_at > registered_at` exactly —
  equality fails. This closes the cute backdoor where a writer registers
  and "observes" at the same millisecond.
- **Boundary-only typed validation.** `probability ∈ [0,1]` and
  `label ∈ {0,1}` are checked at the pydantic boundary. Other keys in the
  prediction dict are preserved untyped so researchers can stash
  experiment metadata without schema churn.
- **Abstract DAO + concrete SQLite impl.** Postgres or any other backend
  is a drop-in subclass; the harness API never sees the storage choice.

### Production hardening

- **Extended calibration metrics.** Beyond Brier and ECE: `log_loss`,
  `accuracy_at_0_5`, `max_calibration_error` (worst-bin equivalent of ECE),
  and `num_realized_positives` / `num_realized_negatives` for sanity
  checks on small windows.
- **Dataset-scoped reports.** `calibration_report(model_id, time_window,
  dataset_hash=None)` lets you slice the report to one dataset so a
  drift event on one feed doesn't contaminate the headline number.
- **`prediction-harness` CLI** with `register / record / report`
  subcommands and JSON in/out — usable from shell or pipeline without
  touching Python.
- **16-thread concurrent-registration test.** Sixteen workers race to
  register the same prediction id; exactly one wins, the other fifteen
  raise `ImmutablePredictionError`. Proves the `UNIQUE` race is handled
  correctly under load.
- **Hypothesis stateful random walk.** A bundle-based state machine
  randomly registers, records, and reports against an in-memory shadow
  model; every transition checks that the harness and the shadow agree
  on counts, totals, and immutability.

---

## Task B — Adversarial Input Detector

**Problem.** Inputs arrive at the institute from many sources, some
adversarial. The detector has to flag prompt-injection, goal corruption,
and directionality-reversed reasoning *cheaply* (the perimeter has a tight
latency budget) and *interpretably* (researchers need to know which
detector tripped).

### Smart approaches

- **Three detectors instead of one big classifier — defense in depth.**
  Each detector has a different failure mode, so an attacker has to defeat
  all three simultaneously to slip through:
  1. **Directionality** — regex from JSON config; designated `hard_block`,
     so any trigger short-circuits to `BLOCK`. Zero false positives by
     construction, brittle to rephrasing.
  2. **Anomaly** — Mahalanobis distance on TF-IDF + TruncatedSVD
     embeddings, fit only on the benign corpus. Model-free, catches novel
     "weird" inputs that the regex misses.
  3. **Classifier** — class-balanced logistic regression on the same
     embeddings. Learns subtle adversarial patterns from labels.
- **Cheap encoder, not a transformer.** `TfidfVectorizer(word + bigrams)`
  feeding a 64-component `TruncatedSVD`. Sub-millisecond per call. The
  perimeter is hot-path and can't afford a 200ms LLM round-trip per query.
- **Aggregation lives in YAML, not in code.** `aggregation.yaml` controls
  weights, thresholds, and `hard_block` flags. Tuning the policy doesn't
  require a code change or redeploy.
- **Corpus designed to exercise specific failure modes** — 61 benign
  vs 61 adversarial examples covering directionality reversal, goal
  corruption, and priority injection. Small, but each axis is represented.

### Production hardening

- **Per-source thresholds.** `public_submission` is stricter than
  `internal_researcher`; the `InputSource` parameter actually gates which
  threshold table is loaded. Configured in `aggregation.yaml`
  (`source_overrides`).
- **`screen_batch(inputs)` vectorized path** — one embedding forward
  pass shared across N payloads. Linear-scan amortized cost; matters
  when an upstream system flushes a batch.
- **5-fold stratified CV at training**, with the encoder refit *inside*
  each fold so there is no leakage. ROC AUC is computed via
  Mann–Whitney U on the held-out fold scores.
- **Artifact metadata in the joblib bundle** — `bundle_version`,
  `trained_at`, `sklearn_version`, corpus SHA-256s, and CV mean/std.
  Makes "which model is in production" answerable from the artifact alone.
- **`adv-screen-check` CLI** for one-shot screening from the shell.
- **4 Hypothesis property tests** (300 examples each): verdict is always
  one of the allowed values; the hard-block invariant holds; thresholds
  are monotone in subscores; and `public_submission` strictness *dominates*
  `internal_researcher` strictness across every possible subscore triple.

**Measured under strictest tier (`public_submission`):**
FP rate 3.28% (target <5%) · recall 100% (target >90%) · p99 latency
0.65 ms (target <500 ms).

---

## Task C — Priority Scheduler with Pre-emption

**Problem.** Schedule mixed-priority compute workloads with a finite
budget. Higher-priority jobs must pre-empt lower-priority ones safely
(checkpoint, suspend, resume later) without ever inverting priority and
without dropping work to crashes inside a user-supplied workload.

### Smart approaches

- **Discrete-time `tick()` simulation, not real threads.** Tests are
  deterministic and reproducible; race conditions don't "pass on Tuesdays".
  Real concurrency is a separate concern, addressed by a pluggable runtime
  adapter (see hardening below).
- **Four priority tiers in YAML** (`critical / high / standard /
  opportunistic`) with explicit `preemptible_by` relations — the policy
  is data, not code.
- **`Workload` is a Protocol.** Callers supply `advance / checkpoint /
  restore / is_complete`. The scheduler doesn't know or care what's
  inside a checkpoint — opaque bytes from its perspective.
- **Append-only audit log** in memory and on disk (JSONL). Every
  admission, pre-emption, resume, completion, and failure is one line of
  JSON, in order. Replayable.
- **Two-wave stress design.** The stress harness submits jobs in two
  waves so pre-emption actually fires: the first wave runs for 3 ticks
  before a higher-priority second wave arrives. Avoids the common bug
  where a stress test "passes" because pre-emption was never exercised.

### Production hardening

- **`JobStatus.FAILED` + `WorkloadError`.** Every callback into user
  code is wrapped in try/except; a buggy workload transitions to FAILED
  with the exception captured, and the scheduler keeps running. Foreign
  code can never crash the scheduler.
- **Checkpoint cleanup on every terminal transition**
  (`COMPLETED / CANCELLED / FAILED`). No orphan checkpoint files
  accumulate on disk over long-running operation.
- **Real-time `PriorityInversionError` guard** in `_admit` and `_resume`.
  Defense in depth on top of priority-ordered fill: even if a future
  refactor reorders the fill logic, the invariant check catches it
  immediately rather than at audit-replay time.
- **Pluggable `RuntimeAdapter`.** `SynchronousRuntime` (default) keeps
  the brain deterministic and testable; `ThreadedRuntime` parallelizes
  the per-tick `advance()` calls without touching scheduler bookkeeping.
  This is the "brain / body" split — the same brain wires into Ray,
  Slurm, or a Kubernetes controller with a localized adapter change.
- **100-seed stress sweep.** `scheduler-stress-sweep --seeds 100`
  exercises 5000 random jobs (50 per seed) and confirms all three
  invariants hold every time.
- **Hypothesis stateful random-walk** with a `Bundle` of submitted
  handles. Per-step invariants (budget, slots, accounting) plus global
  invariants (every yielded job is later resumed *or* cancelled).

**Stress run with seed 42:** 50 workloads, 725/1521 credits used,
1 pre-emption + 1 resume, zero priority inversions.
**Stress sweep (100 seeds):** 5000/5000 completed, 108 pre-emptions,
0 priority inversions.

---

## Task D — External Evidence Client

**Problem.** Five real biomedical APIs (PubMed, ClinicalTrials.gov,
FAERS, NHANES, STRING-DB), each with its own quirks, must be reachable
through one client that *never raises* — every error becomes a typed
result the caller can branch on.

### Smart approaches

- **Discriminated union `Success | Failure`** with
  `status: Literal["success", "failure"]` as the discriminator. Pydantic
  routes deserialization correctly inside a parent model; callers
  branch with `isinstance(r, Success)` or `r.ok()`. Network errors,
  4xx, 5xx, timeouts, and parse failures all map to `Failure` — they
  never propagate as exceptions.
- **Shared `HttpRunner` for cross-cutting concerns.** Retry policy,
  caching, metrics, circuit breaker, and pydantic validation all sit in
  one place. Each source file ends up roughly 30 lines — just URL
  construction and parameter shaping.
- **VCR cassettes recorded against real APIs.** Tests run offline by
  default with the recorded HTTP exchanges, so CI is hermetic.
  `EVIDENCE_LIVE=1` gates 5 real-API integration tests for use when you
  *want* to validate against live upstream.
- **Targeted retry set.** Only transient statuses retry — 429 (rate
  limit) and 5xx (502/503/504 by default) with `Retry-After` honored.
  4xx caller errors do not retry; 500 is configurable but off by
  default because it's ambiguous.

### Production hardening

- **Per-source `CircuitBreaker`.** 5 consecutive failures → OPEN for 30s
  → HALF_OPEN probe → close (success) or re-open (failure). While open,
  calls fail fast as `Failure(message="circuit-open")` so a downstream
  agent doesn't burn latency on a known-dead source.
- **`InMemoryMetrics` + `NullMetrics`.** Per-source counters keyed by
  outcome and status code, plus latency histograms (count/sum/min/max/
  mean). Reachable as `client.metrics` for inspection or scraping.
- **Thread-safe `TTLCache`** with a `threading.Lock` guarding
  get/set/evict so concurrent agents share cache results safely.
- **Per-source pydantic response schemas** (`response_types.py`).
  Upstream shape drift surfaces as `Failure("parse error: …")` instead
  of a `KeyError` deep inside caller code.
- **5xx retry alongside 429.** `RetryPolicy.retry_on_status` is
  configurable per source, so you can tune retry behavior to what each
  upstream actually does well under load.

---

## Task E — Synthetic Cohort Generator

**Problem.** Generate fake-but-statistically-real patient cohorts for
testing pipelines that touch real patient data. The output has to match
declared marginals, correlations, subtype prevalences, and missingness
patterns — and be byte-identical across runs for the same seed.

### Smart approaches

- **Gaussian copula** to separate the *correlation structure*
  (multivariate normal, Cholesky factor) from the *marginal
  distributions* (arbitrary). Uncorrelated noise is rotated through the
  Cholesky factor, then each margin is pushed through its inverse CDF.
- **Analytical Pearson calibration** for non-linear marginals.
  Normal × normal preserves Pearson exactly; anything with `exp()`
  deforms it. The generator solves closed-form inversions for the two
  cases we care about (normal–lognormal and lognormal–lognormal) so
  realized correlations land within ±0.05 of target without numerical
  search.
- **Stratified subtype sampling** via the largest-remainder method.
  Pure multinomial sampling has natural variance √(p(1−p)/n) that
  exceeds ±2pp at p≈0.5, n=1000 in ~20% of seeds; pre-allocating
  `round(prevalence_i · n)` and reconciling rounding to total = n makes
  proportions exact.
- **Reproducibility by construction.** A single
  `np.random.default_rng(seed)` drives all randomness in a fixed order,
  and CSV bytes are byte-identical for `(config, n, seed)`. KS,
  chi-square, Pearson, and proportion tests all run at α=0.05.

### Production hardening

- **Biomarker profiles applied per-subtype.** Each subtype declares
  mean shifts on the features it cares about; shifts are linear so
  they preserve the correlation structure set up by the copula.
- **`MissingnessSpec`.** MCAR (uniform) or MAR (`depends_on` shifts
  the missingness probability based on whether the dependent feature
  is in the upper or lower half). Per-feature, per-rate.
- **`generate_chunks(config, total_n, chunk_size, seed)`.** Streaming
  generation for `n` larger than memory; the RNG state is advanced
  consistently across chunks so the concatenated output is identical
  to a single-shot generation.
- **`cohort-gen` CLI** with `--out` for CSV and `--provenance` for a
  sidecar JSON manifest carrying the config SHA-256, an inline copy of
  the config, the generator version, and the seed. Reproducible runs
  are traceable from the output alone.
- **Hypothesis-fuzzed random-config tests.** 40 random `DiseaseConfig`s
  checked against four invariants: shape, subtype proportions within
  ±2pp, mean drift on shifted features, byte-identical reproducibility.

---

## Dependencies

The combined venv installs:

- `pydantic`, `numpy`, `pandas`, `scipy`
- `sqlalchemy` (Task A)
- `scikit-learn`, `pyyaml`, `joblib` (Tasks B, C)
- `requests`, `vcrpy` (Task D)
- `jupyter`, `matplotlib` (Task E demo notebook — optional)
- `pytest`, `hypothesis` (all tasks)

No external services are required to run the test suites; Task D
ships pre-recorded cassettes so the run is fully offline.
