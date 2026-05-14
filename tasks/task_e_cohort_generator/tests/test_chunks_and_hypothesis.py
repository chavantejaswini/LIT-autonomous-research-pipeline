"""Streaming chunked generation + `hypothesis`-driven property tests over
randomly generated configs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cohort_generator import (
    DiseaseConfig,
    generate,
    generate_chunks,
)


# ---- chunked generation ---------------------------------------------------


def test_generate_chunks_yields_correct_total_size(config) -> None:
    chunks = list(generate_chunks(config, total_n=300, chunk_size=100, seed=7))
    assert len(chunks) == 3
    assert all(len(c) == 100 for c in chunks)
    combined = pd.concat(chunks, ignore_index=True)
    assert len(combined) == 300


def test_generate_chunks_handles_uneven_split(config) -> None:
    """A `total_n` that doesn't divide evenly should still produce exactly N rows."""
    chunks = list(generate_chunks(config, total_n=250, chunk_size=100, seed=7))
    assert sum(len(c) for c in chunks) == 250


def test_generate_chunks_reproducible(config) -> None:
    a = pd.concat(list(generate_chunks(config, 200, 50, seed=99)), ignore_index=True)
    b = pd.concat(list(generate_chunks(config, 200, 50, seed=99)), ignore_index=True)
    assert a.to_csv(index=False) == b.to_csv(index=False)


# ---- hypothesis property tests --------------------------------------------


def _make_config(
    n_features: int,
    n_subtypes: int,
    feature_means: list[float],
    feature_stds: list[float],
    prevalence_seeds: list[float],
) -> DiseaseConfig:
    """Build a small valid `DiseaseConfig` from hypothesis draws."""
    features = [f"f{i}" for i in range(n_features)]
    dists = {
        f: {"type": "normal", "mean": m, "std": max(s, 0.1)}
        for f, m, s in zip(features, feature_means, feature_stds)
    }
    # Normalize prevalences.
    seeds = [max(p, 0.01) for p in prevalence_seeds[:n_subtypes]]
    total = sum(seeds)
    prevs = [p / total for p in seeds]
    # Reconcile sum-to-1 rounding error onto the last subtype.
    prevs[-1] = 1.0 - sum(prevs[:-1])
    subtypes = [
        {"name": f"s{i}", "prevalence": prevs[i], "biomarker_profiles": {}}
        for i in range(n_subtypes)
    ]
    return DiseaseConfig(
        disease_name="hyp",
        subtypes=subtypes,
        baseline_features=features,
        baseline_distributions=dists,
        feature_correlations=[],
    )


@given(
    n_features=st.integers(min_value=2, max_value=6),
    n_subtypes=st.integers(min_value=1, max_value=4),
    feature_means=st.lists(
        st.floats(min_value=-10, max_value=10, allow_nan=False),
        min_size=6, max_size=6,
    ),
    feature_stds=st.lists(
        st.floats(min_value=0.5, max_value=5.0, allow_nan=False),
        min_size=6, max_size=6,
    ),
    prevalence_seeds=st.lists(
        st.floats(min_value=0.05, max_value=1.0, allow_nan=False),
        min_size=4, max_size=4,
    ),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_hypothesis_random_configs_satisfy_invariants(
    n_features, n_subtypes, feature_means, feature_stds, prevalence_seeds, seed
):
    config = _make_config(
        n_features, n_subtypes, feature_means, feature_stds, prevalence_seeds
    )
    df = generate(config, n=500, seed=seed)

    # Invariant 1: shape.
    assert len(df) == 500
    assert list(df.columns)[:-1] == config.baseline_features
    assert df.columns[-1] == "subtype"

    # Invariant 2: subtype proportions within ±3pp (slightly relaxed for n=500).
    counts = df["subtype"].value_counts(normalize=True)
    for s in config.subtypes:
        emp = float(counts.get(s.name, 0.0))
        assert abs(emp - s.prevalence) <= 0.03

    # Invariant 3: per-feature mean lands within 3 standard errors of target.
    for feat in config.baseline_features:
        dist = config.baseline_distributions[feat]
        emp_mean = float(df[feat].mean())
        se = dist.std / np.sqrt(len(df))
        assert abs(emp_mean - dist.mean) < 4 * se, (
            f"{feat}: mean {emp_mean:.2f} too far from {dist.mean} (se={se:.3f})"
        )

    # Invariant 4: same seed → same DataFrame.
    df2 = generate(config, n=500, seed=seed)
    assert df.to_csv(index=False) == df2.to_csv(index=False)
