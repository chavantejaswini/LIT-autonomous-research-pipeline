"""Schema-level checks on `DiseaseConfig`."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from cohort_generator.config import (
    BinaryDist,
    CategoricalDist,
    CorrelationPair,
    DiseaseConfig,
    LogNormalDist,
    NormalDist,
    Subtype,
)


def _minimal_kwargs(**overrides):
    base = dict(
        disease_name="x",
        subtypes=[Subtype(name="a", prevalence=1.0, biomarker_profiles={})],
        baseline_features=["x"],
        baseline_distributions={"x": NormalDist(type="normal", mean=0, std=1)},
        feature_correlations=[],
    )
    base.update(overrides)
    return base


def test_prevalences_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        DiseaseConfig(
            **_minimal_kwargs(
                subtypes=[
                    Subtype(name="a", prevalence=0.3, biomarker_profiles={}),
                    Subtype(name="b", prevalence=0.3, biomarker_profiles={}),
                ]
            )
        )


def test_missing_distribution_for_feature_raises() -> None:
    with pytest.raises(ValueError):
        DiseaseConfig(
            **_minimal_kwargs(
                baseline_features=["x", "y"],
                baseline_distributions={"x": NormalDist(type="normal", mean=0, std=1)},
            )
        )


def test_correlation_with_self_raises() -> None:
    with pytest.raises(ValueError):
        DiseaseConfig(
            **_minimal_kwargs(
                feature_correlations=[
                    CorrelationPair(feature_a="x", feature_b="x", pearson_r=0.5)
                ]
            )
        )


def test_pearson_r_outside_range_raises() -> None:
    with pytest.raises(ValidationError):
        CorrelationPair(feature_a="a", feature_b="b", pearson_r=1.2)


def test_categorical_probabilities_normalize() -> None:
    d = CategoricalDist(
        type="categorical", categories=["x", "y"], probabilities=[2.0, 8.0]
    )
    assert d.probabilities == [0.2, 0.8]
