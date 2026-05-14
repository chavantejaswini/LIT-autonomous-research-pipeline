"""Missingness specs are honored at the expected rate.

For MCAR: empirical NaN rate matches the spec within ±2pp at n=2000.
For MAR (depends_on a continuous feature): the upper-half group has a
higher NaN rate than the lower-half group, by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cohort_generator import (
    DiseaseConfig,
    MissingnessSpec,
    NormalDist,
    generate,
    load_config,
)
from cohort_generator.config import LogNormalDist
from cohort_generator.config import BinaryDist
from cohort_generator.config import CategoricalDist

N = 2000
SEED = 42


def test_no_missingness_when_unspecified() -> None:
    """A config without `missingness` produces no NaNs."""
    config = DiseaseConfig(
        disease_name="x",
        subtypes=[{"name": "a", "prevalence": 1.0, "biomarker_profiles": {}}],
        baseline_features=["x"],
        baseline_distributions={"x": {"type": "normal", "mean": 0, "std": 1}},
        feature_correlations=[],
    )
    df = generate(config, n=N, seed=SEED)
    assert df["x"].isna().sum() == 0


def test_mcar_rate_within_2pp() -> None:
    config = DiseaseConfig(
        disease_name="x",
        subtypes=[{"name": "a", "prevalence": 1.0, "biomarker_profiles": {}}],
        baseline_features=["x"],
        baseline_distributions={
            "x": {
                "type": "normal", "mean": 0, "std": 1,
                "missingness": {"rate": 0.20, "mechanism": "mcar"},
            },
        },
        feature_correlations=[],
    )
    df = generate(config, n=N, seed=SEED)
    empirical = df["x"].isna().mean()
    assert abs(empirical - 0.20) < 0.02


def test_mar_higher_missing_in_upper_half_of_depends_on() -> None:
    """When `mechanism=mar` and `depends_on=age`, rows in the upper-half
    of `age` should miss more often than rows in the lower-half."""
    config = DiseaseConfig(
        disease_name="x",
        subtypes=[{"name": "a", "prevalence": 1.0, "biomarker_profiles": {}}],
        baseline_features=["age", "y"],
        baseline_distributions={
            "age": {"type": "normal", "mean": 60, "std": 10},
            "y": {
                "type": "normal", "mean": 0, "std": 1,
                "missingness": {"rate": 0.20, "mechanism": "mar", "depends_on": "age"},
            },
        },
        feature_correlations=[],
    )
    df = generate(config, n=N, seed=SEED)
    median_age = df["age"].median()
    upper_miss = df[df["age"] > median_age]["y"].isna().mean()
    lower_miss = df[df["age"] <= median_age]["y"].isna().mean()
    assert upper_miss > lower_miss + 0.05, (
        f"MAR should mask more in upper half: upper={upper_miss:.3f}, lower={lower_miss:.3f}"
    )


def test_missingness_preserves_reproducibility() -> None:
    """Same (config, n, seed) → byte-identical DataFrame even with missingness."""
    config = DiseaseConfig(
        disease_name="x",
        subtypes=[{"name": "a", "prevalence": 1.0, "biomarker_profiles": {}}],
        baseline_features=["x"],
        baseline_distributions={
            "x": {
                "type": "normal", "mean": 0, "std": 1,
                "missingness": {"rate": 0.15, "mechanism": "mcar"},
            },
        },
        feature_correlations=[],
    )
    df1 = generate(config, n=N, seed=SEED)
    df2 = generate(config, n=N, seed=SEED)
    assert df1.to_csv(index=False) == df2.to_csv(index=False)


def test_example_config_missingness_observed(config: DiseaseConfig) -> None:
    """For any feature in the example configs with a missingness spec, the
    empirical NaN rate should land near the configured rate."""
    df = generate(config, n=N, seed=SEED)
    for feat in config.baseline_features:
        dist = config.baseline_distributions[feat]
        spec = getattr(dist, "missingness", None)
        if spec is None or spec.rate == 0.0:
            continue
        empirical = df[feat].isna().mean()
        # MCAR is tight; MAR is wider because of the high/low split.
        tol = 0.03 if spec.mechanism == "mcar" else 0.08
        assert abs(empirical - spec.rate) < tol, (
            f"feature {feat} mechanism={spec.mechanism}: empirical "
            f"{empirical:.3f} far from spec {spec.rate} (tol {tol})"
        )
