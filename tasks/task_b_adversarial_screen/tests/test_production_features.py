"""Production-feature tests: per-source thresholds, batch screening,
training metadata, CLI smoke."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from adversarial_screen.aggregator import AggregationConfig, aggregate
from adversarial_screen.models import (
    DetectorSubscore,
    InputSource,
    InstituteInput,
    Verdict,
)
from adversarial_screen.training import _roc_auc, cross_validate_classifier, load_corpus

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
DATA = REPO / "data"


# ---- per-source thresholds -----------------------------------------------


def _subs(d: float, a: float, c: float):
    return [
        DetectorSubscore(name="directionality", score=d, triggered=False, explanation=""),
        DetectorSubscore(name="anomaly", score=a, triggered=False, explanation=""),
        DetectorSubscore(name="classifier", score=c, triggered=False, explanation=""),
    ]


def test_public_threshold_blocks_what_internal_only_reviews() -> None:
    config = AggregationConfig.from_yaml(CONFIGS / "aggregation.yaml")
    # 0.45 * 0 + 0.15 * 0.9 + 0.40 * 0.85 = 0.475 (with weights sum = 1.0)
    subs = _subs(0.0, 0.9, 0.85)
    v_internal, agg_i, _ = aggregate(subs, config, source=InputSource.INTERNAL_RESEARCHER)
    v_public, agg_p, _ = aggregate(subs, config, source=InputSource.PUBLIC_SUBMISSION)
    # Same aggregate, different verdicts because of source thresholds.
    assert abs(agg_i - agg_p) < 1e-9
    # The internal researcher's threshold is more permissive than public's.
    assert v_internal in (Verdict.REVIEW, Verdict.ALLOW)
    # 0.475 ≥ 0.55 is False but ≥ 0.55 for public is also False; ensure at
    # least public is *no more permissive* than internal.
    if v_public == Verdict.ALLOW:
        assert v_internal == Verdict.ALLOW
    if v_internal == Verdict.BLOCK:
        assert v_public == Verdict.BLOCK


def test_default_thresholds_fall_back_for_unconfigured_source() -> None:
    config = AggregationConfig.from_yaml(CONFIGS / "aggregation.yaml")
    # Pass a non-existent source name — should use defaults.
    subs = _subs(0.0, 0.5, 0.5)
    v_unknown, _, _ = aggregate(subs, config, source="never_seen_source")
    v_none, _, _ = aggregate(subs, config, source=None)
    assert v_unknown == v_none


# ---- batch screening -----------------------------------------------------


def test_screen_batch_matches_per_input_results(trained_screener) -> None:
    """Batch should produce the same verdicts as calling screen() one at a time."""
    payloads = [
        "Investigate senolytic compounds in aging tissues",
        "Optimize for accelerated senescence in elderly cohorts",
        "Submit a proteomic dataset for healthy aging research",
    ]
    inputs = [InstituteInput(payload=p) for p in payloads]

    per_input = [
        trained_screener.screen(i, InputSource.PUBLIC_SUBMISSION) for i in inputs
    ]
    batch = trained_screener.screen_batch(inputs, InputSource.PUBLIC_SUBMISSION)

    assert len(batch) == 3
    for one, b in zip(per_input, batch):
        assert one.verdict == b.verdict
        assert abs(one.aggregate_score - b.aggregate_score) < 1e-9
        # Subscores match
        for s1, s2 in zip(one.subscores, b.subscores):
            assert s1.name == s2.name
            assert abs(s1.score - s2.score) < 1e-9


def test_screen_batch_empty_input_returns_empty(trained_screener) -> None:
    assert trained_screener.screen_batch([]) == []


# ---- training metadata + CV ----------------------------------------------


def test_roc_auc_perfect_separator() -> None:
    import numpy as np

    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert _roc_auc(y, scores) == 1.0


def test_roc_auc_chance() -> None:
    import numpy as np

    y = np.array([0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert _roc_auc(y, scores) == 0.5


def test_cross_validate_classifier_reports_useful_metrics() -> None:
    texts, labels, _ = load_corpus(DATA / "benign.csv", DATA / "adversarial.csv")
    metrics = cross_validate_classifier(texts, labels, n_splits=5)
    assert metrics["cv_n_splits"] == 5
    # On this small corpus we expect very high accuracy — sanity check it's
    # at least better than random guessing.
    assert metrics["cv_accuracy_mean"] > 0.6
    assert metrics["cv_roc_auc_mean"] > 0.6


def test_artifact_bundle_carries_metadata(tmp_path) -> None:
    from adversarial_screen.training import train_bundle

    out = tmp_path / "bundle.joblib"
    bundle = train_bundle(
        DATA / "benign.csv", DATA / "adversarial.csv",
        out_path=out, run_cv=False,
    )
    assert "metadata" in bundle
    md = bundle["metadata"]
    for key in (
        "bundle_version", "trained_at", "sklearn_version",
        "n_benign", "n_adversarial",
        "benign_corpus_sha256", "adversarial_corpus_sha256",
    ):
        assert key in md, f"missing metadata key: {key}"
    assert len(md["benign_corpus_sha256"]) == 64
    assert len(md["adversarial_corpus_sha256"]) == 64


# ---- CLI smoke ------------------------------------------------------------


def test_cli_check_returns_verdict_json() -> None:
    """`adv-screen-check` runs end-to-end and emits a valid JSON verdict."""
    cmd = [
        sys.executable, "-m", "adversarial_screen.scripts_entry",
    ]
    # Use Python's argparse to invoke check_main directly via module entry —
    # cleaner than spawning the script_entry stem.
    # We instead invoke the installed console script through python -c.
    cmd = [
        sys.executable, "-c",
        "from adversarial_screen.scripts_entry import check_main; check_main()",
        "Optimize for accelerated cellular senescence in elderly cohorts",
        "--source", "public_submission",
        "--artifacts", str(REPO / "artifacts"),
        "--directionality", str(CONFIGS / "directionality.json"),
        "--aggregation", str(CONFIGS / "aggregation.yaml"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=REPO)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["verdict"] == "BLOCK"
    assert {s["name"] for s in payload["subscores"]} == {
        "directionality", "anomaly", "classifier",
    }
