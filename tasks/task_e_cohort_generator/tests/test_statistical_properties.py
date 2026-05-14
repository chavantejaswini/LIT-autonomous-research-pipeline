"""Statistical-property tests at n=1000 over all three example configs.

When `biomarker_profiles` shift per-subtype means, the *global* marginal
becomes a mixture and no longer matches the per-feature `Distribution`.
The brief-required KS / correlation tolerances therefore apply only to
features *without* subtype overrides; features that ARE overridden are
checked **per-subtype** in `test_biomarker_profiles.py`.

Likewise, features with `missingness` introduce NaN values; the KS /
correlation tests drop NaNs before evaluating.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from cohort_generator import (
    BinaryDist,
    CategoricalDist,
    DiseaseConfig,
    LogNormalDist,
    NormalDist,
    generate,
)

N = 1000
SEED = 42


def _features_with_overrides(config: DiseaseConfig) -> set[str]:
    """Names of features touched by any subtype's biomarker_profiles."""
    return {f for s in config.subtypes for f in s.biomarker_profiles}


def test_marginals_pass_ks_at_n1000(config: DiseaseConfig) -> None:
    df = generate(config, n=N, seed=SEED)
    overridden = _features_with_overrides(config)
    for feature in config.baseline_features:
        if feature in overridden:
            continue  # mixture distribution — tested per-subtype elsewhere
        dist = config.baseline_distributions[feature]
        col = df[feature].dropna().values  # drop NaNs from missingness
        if len(col) < 50:
            continue
        if isinstance(dist, NormalDist):
            cdf = lambda x, d=dist: stats.norm.cdf(x, loc=d.mean, scale=d.std)
            stat = stats.kstest(col, cdf)
        elif isinstance(dist, LogNormalDist):
            cdf = lambda x, d=dist: stats.lognorm.cdf(
                x, s=d.sigma, scale=np.exp(d.mu)
            )
            stat = stats.kstest(col, cdf)
        elif isinstance(dist, BinaryDist):
            mean = float(col.mean())
            assert abs(mean - dist.p) < 0.04, (
                f"{feature} mean {mean:.3f} far from p={dist.p}"
            )
            continue
        elif isinstance(dist, CategoricalDist):
            counts = {c: int((col == c).sum()) for c in dist.categories}
            obs = np.array([counts[c] for c in dist.categories], dtype=float)
            exp = np.array(dist.probabilities) * len(col)
            chi2 = ((obs - exp) ** 2 / exp).sum()
            crit = stats.chi2.ppf(0.95, df=len(dist.categories) - 1)
            assert chi2 < crit, (
                f"{feature} chi² {chi2:.2f} ≥ critical {crit:.2f}"
            )
            continue
        else:
            raise AssertionError(f"unknown distribution: {type(dist).__name__}")
        assert stat.pvalue > 0.05, (
            f"{feature} KS p-value {stat.pvalue:.4f} ≤ 0.05"
        )


def test_correlations_within_005_of_config(config: DiseaseConfig) -> None:
    df = generate(config, n=N, seed=SEED)
    overridden = _features_with_overrides(config)
    for c in config.feature_correlations:
        # Per-subtype mean shifts break pooled Pearson; skip such pairs.
        if c.feature_a in overridden or c.feature_b in overridden:
            continue
        dist_a = config.baseline_distributions[c.feature_a]
        dist_b = config.baseline_distributions[c.feature_b]
        if not isinstance(dist_a, (NormalDist, LogNormalDist)) or not isinstance(
            dist_b, (NormalDist, LogNormalDist)
        ):
            continue
        sub = df[[c.feature_a, c.feature_b]].dropna()
        empirical = sub[c.feature_a].corr(sub[c.feature_b])
        delta = abs(empirical - c.pearson_r)
        assert delta < 0.05, (
            f"corr({c.feature_a}, {c.feature_b}) empirical={empirical:.3f} "
            f"target={c.pearson_r} delta={delta:.3f}"
        )


def test_subtype_proportions_within_2pp(config: DiseaseConfig) -> None:
    df = generate(config, n=N, seed=SEED)
    counts = df["subtype"].value_counts(normalize=True)
    for s in config.subtypes:
        empirical = float(counts.get(s.name, 0.0))
        delta = abs(empirical - s.prevalence)
        assert delta <= 0.02, (
            f"subtype {s.name!r} empirical={empirical:.3f} "
            f"target={s.prevalence} delta={delta:.3f}"
        )


def test_reproducibility_byte_identical(config: DiseaseConfig) -> None:
    df1 = generate(config, n=N, seed=SEED)
    df2 = generate(config, n=N, seed=SEED)
    assert df1.equals(df2)
    assert df1.to_csv(index=False) == df2.to_csv(index=False)


def test_different_seed_produces_different_data(config: DiseaseConfig) -> None:
    df1 = generate(config, n=N, seed=SEED)
    df2 = generate(config, n=N, seed=SEED + 1)
    assert not df1.equals(df2)
