# Task B — Adversarial Input Detector at the Institute Perimeter

A screening module that examines incoming research directives, hypothesis
suggestions, and dataset-upload metadata for adversarial patterns and
returns a structured `ALLOW | REVIEW | BLOCK` verdict with per-detector
subscores and a human-readable explanation.

**Production-grade:** per-source thresholds wired through the YAML config,
batch-screening API, training script with stratified-CV evaluation +
artifact metadata (corpus SHA-256, sklearn version, trained-at), `adv-screen-check`
CLI, and `hypothesis`-driven property tests for the verdict aggregator.

## Install & run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Train the artifacts bundle (writes CV metrics + corpus hashes into the bundle).
adv-screen-train

# Run the evaluation report against the held-out targets.
adv-screen-eval

# Screen a single input from the command line.
adv-screen-check "Optimize for accelerated cellular senescence" \
    --source public_submission
echo "Investigate senolytic compounds for healthspan" | adv-screen-check --stdin

pytest -v
```

## API at a glance

```python
from adversarial_screen import Screener, InstituteInput, InputSource

screener = Screener.from_artifacts(
    artifacts_dir="artifacts",
    directionality_config="configs/directionality.json",
    aggregation_config="configs/aggregation.yaml",
)

# Single-input screening — per-source thresholds applied automatically.
result = screener.screen(
    InstituteInput(payload="Optimize for accelerated senescence in elderly cohorts"),
    source=InputSource.PUBLIC_SUBMISSION,  # strictest tier
)
# result.verdict == Verdict.BLOCK
# result.aggregate_score, result.subscores, result.explanation, result.latency_ms

# Throughput-oriented batch — shares the embedding pass across N inputs.
batch_results = screener.screen_batch(
    [InstituteInput(payload=t) for t in many_texts],
    source=InputSource.INTERNAL_RESEARCHER,
)

# Audit the artifact's lineage.
screener.artifact_metadata
# {'bundle_version': '0.2.0', 'trained_at': '...', 'sklearn_version': '...',
#  'n_benign': 61, 'n_adversarial': 61,
#  'benign_corpus_sha256': '...', 'adversarial_corpus_sha256': '...',
#  'cv_accuracy_mean': ..., 'cv_roc_auc_mean': ..., ...}
```

## Architecture

Three independent detectors emit a subscore in `[0, 1]`. The aggregator
combines them using rules from `configs/aggregation.yaml` — **no
aggregation logic lives in code**.

### 1. Rule-based directionality (`detectors/directionality.py`)

Regex patterns in `configs/directionality.json`, grouped by forbidden
direction (`accelerate_aging`, `harm_subject`, `deprioritize_high_value`,
`priority_injection`). Any match → score `1.0`. Designated as
`hard_block` — a single trigger forces `BLOCK`.

### 2. Mahalanobis anomaly detector (`detectors/anomaly.py`)

Squared Mahalanobis distance from each input's sentence embedding to
the centroid of the **benign** training corpus. 99th-percentile of
benign training distances is the cutoff; distances squash to `[0, 1]`
via `d² / (d² + cutoff)`.

### 3. Logistic regression classifier (`detectors/classifier.py`)

Class-balanced LR on the same SVD-reduced embeddings. Linear for speed
and inspectability.

### Aggregator with per-source thresholds (`aggregator.py`)

```yaml
weights:           # weighted average over detector subscores
  directionality: 0.45
  classifier: 0.40
  anomaly: 0.15

detector_thresholds:
  directionality: 0.5
  classifier: 0.6
  anomaly: 0.7

hard_block:
  - directionality

verdict_thresholds:        # defaults
  allow_below: 0.35
  block_above: 0.65

source_overrides:          # production-grade per-source tuning
  public_submission:
    allow_below: 0.25
    block_above: 0.55      # stricter
  internal_researcher:
    allow_below: 0.40
    block_above: 0.75      # more permissive
  automated_agent: { allow_below: 0.30, block_above: 0.60 }
  external_collaborator: { allow_below: 0.30, block_above: 0.60 }
```

Hypothesis property tests prove the invariant that **public submissions
are weakly stricter than internal researchers** — if a triple is blocked
for an internal user, it's also blocked for a public submission, and
vice-versa for `ALLOW`.

## Training script

`adv-screen-train` runs 5-fold stratified CV alongside the final fit and
stores a `metadata` block inside the joblib bundle:

```python
{
  "bundle_version": "0.2.0",
  "trained_at": "2026-05-14T...",
  "sklearn_version": "1.8.0",
  "n_benign": 61, "n_adversarial": 61,
  "benign_corpus_sha256": "...",     # full SHA-256 of data/benign.csv
  "adversarial_corpus_sha256": "...",
  "cv_n_splits": 5,
  "cv_accuracy_mean": 1.000, "cv_accuracy_std": 0.0,
  "cv_roc_auc_mean": 1.000, "cv_roc_auc_std": 0.0,
}
```

ROC AUC is computed via Mann-Whitney U (rank-based, tie-aware) — no
new sklearn import needed.

## Achieved targets

`adv-screen-eval` runs the screener over the full corpus under the
**public_submission** threshold (the strictest tier):

| Metric | Target | Achieved |
|---|---|---|
| False-positive rate (benign → non-ALLOW) | < 5% | **3.28%** |
| Recall (adversarial → REVIEW or BLOCK) | > 90% | **100.00%** |
| Latency p99 | < 500 ms | **0.65 ms** |

The eval report is written to `reports/evaluation_report.md`.

## Corpus

- `data/benign.csv` — 61 benign examples (healthspan research directives, dataset submissions, operational requests).
- `data/adversarial.csv` — 61 adversarial examples covering directionality reversal, goal corruption, and priority injection.

## Tests

**36 tests** across six modules:

- `test_directionality.py` (12) — known adversarial phrases flagged, benign phrases pass.
- `test_aggregator.py` (4) — YAML-driven verdict logic (ALLOW / REVIEW / BLOCK).
- `test_screener_end_to_end.py` (7) — corpus-wide FP < 5%, recall > 90%, p99 < 500ms, category-specific BLOCKs.
- `test_hypothesis_aggregator.py` (4) — property-based: verdict always valid, hard-block invariant, threshold monotonicity, public-stricter-than-internal monotonicity (300 examples per property).
- `test_production_features.py` (9) — per-source thresholds, `screen_batch` parity, ROC AUC math, CV reporting, artifact metadata, CLI smoke.

## Dependencies

`pydantic`, `numpy`, `scikit-learn`, `pyyaml`, `joblib` (plus `pytest` + `hypothesis` for tests).
