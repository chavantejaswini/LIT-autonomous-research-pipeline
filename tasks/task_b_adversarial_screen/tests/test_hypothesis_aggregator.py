"""Property-based tests for the verdict aggregator.

Hypothesis generates arbitrary (directionality, anomaly, classifier)
subscore triples and verifies the aggregator's invariants:

  * Output verdict is always one of ALLOW / REVIEW / BLOCK.
  * Aggregate score is always in [0, 1].
  * If directionality fires above its threshold, verdict is BLOCK (hard-block).
  * For non-hard-block triples, verdict mapping respects the configured
    thresholds — strict allow-below, strict block-above.
  * Per-source overrides actually make `public_submission` weakly stricter
    than `internal_researcher`: if a triple is BLOCKed for internal, it
    must also be BLOCKed for public (the other direction need not hold).
"""
from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings, strategies as st

from adversarial_screen.aggregator import AggregationConfig, aggregate
from adversarial_screen.models import DetectorSubscore, InputSource, Verdict

CONFIG = AggregationConfig.from_yaml(
    Path(__file__).resolve().parents[1] / "configs" / "aggregation.yaml"
)


def _sub(name: str, score: float) -> DetectorSubscore:
    return DetectorSubscore(name=name, score=score, triggered=False, explanation="")


def _make(dir_s: float, anom_s: float, clf_s: float):
    return [
        _sub("directionality", dir_s),
        _sub("anomaly", anom_s),
        _sub("classifier", clf_s),
    ]


score = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@given(d=score, a=score, c=score)
@settings(max_examples=300, deadline=None)
def test_verdict_is_always_valid(d, a, c):
    verdict, agg, _ = aggregate(_make(d, a, c), CONFIG)
    assert verdict in (Verdict.ALLOW, Verdict.REVIEW, Verdict.BLOCK)
    assert 0.0 <= agg <= 1.0


@given(d=score, a=score, c=score)
@settings(max_examples=300, deadline=None)
def test_hard_block_when_directionality_triggered(d, a, c):
    """Whenever directionality ≥ its threshold, the verdict must be BLOCK."""
    threshold = CONFIG.detector_thresholds["directionality"]
    verdict, _, _ = aggregate(_make(d, a, c), CONFIG)
    if d >= threshold:
        assert verdict == Verdict.BLOCK


@given(d=score, a=score, c=score)
@settings(max_examples=300, deadline=None)
def test_threshold_monotonicity_for_non_hard_block(d, a, c):
    """For triples that aren't hard-blocked, agg < allow_below ⇒ ALLOW and
    agg ≥ block_above ⇒ BLOCK."""
    threshold = CONFIG.detector_thresholds["directionality"]
    if d >= threshold:
        return  # hard-block path; covered separately
    verdict, agg, _ = aggregate(_make(d, a, c), CONFIG)
    if agg < CONFIG.allow_below:
        assert verdict == Verdict.ALLOW
    elif agg >= CONFIG.block_above:
        assert verdict == Verdict.BLOCK
    else:
        assert verdict == Verdict.REVIEW


@given(d=score, a=score, c=score)
@settings(max_examples=300, deadline=None)
def test_public_submissions_at_least_as_strict_as_internal(d, a, c):
    """For every triple, BLOCK-on-internal implies BLOCK-on-public.
    Equivalently: public's strictness is monotone with respect to internal."""
    pub, _, _ = aggregate(
        _make(d, a, c), CONFIG, source=InputSource.PUBLIC_SUBMISSION
    )
    internal, _, _ = aggregate(
        _make(d, a, c), CONFIG, source=InputSource.INTERNAL_RESEARCHER
    )
    if internal == Verdict.BLOCK:
        assert pub == Verdict.BLOCK
    # And: ALLOW-on-public implies ALLOW-on-internal (strictness in the
    # other direction).
    if pub == Verdict.ALLOW:
        assert internal == Verdict.ALLOW
