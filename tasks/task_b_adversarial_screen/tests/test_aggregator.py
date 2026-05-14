"""Verdict aggregator tests — rules live in YAML, not in code."""
from __future__ import annotations

from pathlib import Path

import pytest

from adversarial_screen.aggregator import AggregationConfig, aggregate
from adversarial_screen.models import DetectorSubscore, Verdict

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "aggregation.yaml"


@pytest.fixture(scope="module")
def config() -> AggregationConfig:
    return AggregationConfig.from_yaml(CONFIG_PATH)


def _sub(name: str, score: float) -> DetectorSubscore:
    return DetectorSubscore(name=name, score=score, triggered=False, explanation="")


def test_all_low_scores_allow(config: AggregationConfig) -> None:
    subs = [_sub("directionality", 0.0), _sub("anomaly", 0.1), _sub("classifier", 0.1)]
    verdict, agg, _ = aggregate(subs, config)
    assert verdict == Verdict.ALLOW
    assert agg < config.allow_below


def test_directionality_match_is_hard_block(config: AggregationConfig) -> None:
    """Directionality is in `hard_block`; any trigger above its threshold blocks."""
    subs = [_sub("directionality", 1.0), _sub("anomaly", 0.0), _sub("classifier", 0.0)]
    verdict, _, _ = aggregate(subs, config)
    assert verdict == Verdict.BLOCK


def test_mid_score_routes_to_review(config: AggregationConfig) -> None:
    # 0.45*0 + 0.15*0.8 + 0.40*0.7 = 0.40 → falls between allow_below and block_above
    subs = [_sub("directionality", 0.0), _sub("anomaly", 0.8), _sub("classifier", 0.7)]
    verdict, agg, _ = aggregate(subs, config)
    assert verdict == Verdict.REVIEW
    assert config.allow_below <= agg < config.block_above


def test_high_classifier_alone_can_block(config: AggregationConfig) -> None:
    # classifier weight 0.40, classifier score 1.0 → contributes 0.40 to weighted avg.
    # We need agg >= block_above (0.65). Bump anomaly too.
    subs = [_sub("directionality", 0.0), _sub("anomaly", 1.0), _sub("classifier", 1.0)]
    verdict, agg, _ = aggregate(subs, config)
    # directionality 0 with weight 0.45, plus anomaly 1.0 × 0.15 + classifier 1.0 × 0.40
    # = 0.55 / 1.0 = 0.55 → REVIEW (between 0.35 and 0.65).
    assert verdict == Verdict.REVIEW
    assert agg == pytest.approx(0.55, abs=1e-6)
