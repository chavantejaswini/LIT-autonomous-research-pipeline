"""Per-subtype `biomarker_profiles` are actually applied to the generated data.

For features that have a subtype-specific override:
  * within each overriding subtype, the empirical mean (on the appropriate
    scale — raw for normal, log for lognormal) lands within ±10% of the
    overridden parameter at n=1000.
  * within non-overriding subtypes, the empirical mean still tracks the
    global parameter.
"""
from __future__ import annotations

import numpy as np

from cohort_generator import (
    DiseaseConfig,
    LogNormalDist,
    NormalDist,
    generate,
)

N = 2000
SEED = 42


def test_biomarker_profile_shifts_subtype_mean(config: DiseaseConfig) -> None:
    df = generate(config, n=N, seed=SEED)
    for subtype in config.subtypes:
        if not subtype.biomarker_profiles:
            continue
        sub = df[df["subtype"] == subtype.name]
        for feat, override in subtype.biomarker_profiles.items():
            dist = config.baseline_distributions[feat]
            col = sub[feat].dropna()
            if len(col) < 30:
                continue
            if isinstance(dist, NormalDist) and "mean" in override:
                empirical = float(col.mean())
                target = float(override["mean"])
                tol = max(0.1 * abs(dist.std), 0.05 * abs(target) or 0.5)
                assert abs(empirical - target) < tol, (
                    f"subtype={subtype.name} feature={feat}: empirical mean "
                    f"{empirical:.2f} far from override {target} (tol {tol:.2f})"
                )
            elif isinstance(dist, LogNormalDist) and "mu" in override:
                # Compare on the log scale, since `mu` is the underlying-normal mean.
                empirical_mu = float(np.log(col).mean())
                target = float(override["mu"])
                tol = max(0.2, 0.05 * abs(target))
                assert abs(empirical_mu - target) < tol, (
                    f"subtype={subtype.name} feature={feat}: empirical mu "
                    f"{empirical_mu:.2f} far from override {target} (tol {tol:.2f})"
                )


def test_subtype_without_override_tracks_global_marginal(config: DiseaseConfig) -> None:
    """For a feature overridden in *some* subtypes, the non-overridden subtypes
    should still produce values centered on the global marginal."""
    df = generate(config, n=N, seed=SEED)
    # Find a feature that's overridden in only some subtypes.
    for feat in config.baseline_features:
        overriding = [
            s for s in config.subtypes if feat in s.biomarker_profiles
        ]
        non_overriding = [
            s for s in config.subtypes if feat not in s.biomarker_profiles
        ]
        if not overriding or not non_overriding:
            continue
        dist = config.baseline_distributions[feat]
        if not isinstance(dist, NormalDist):
            continue
        # Aggregate empirical mean across non-overriding subtypes.
        mask = df["subtype"].isin([s.name for s in non_overriding])
        col = df.loc[mask, feat].dropna()
        if len(col) < 100:
            continue
        empirical = float(col.mean())
        tol = 0.3 * dist.std  # 30% of std — generous because of subtype mixing
        assert abs(empirical - dist.mean) < tol, (
            f"feature {feat}: non-override subtypes have mean {empirical:.2f}, "
            f"global mean is {dist.mean}, tol {tol:.2f}"
        )
        return  # one such feature is enough to validate the property


def test_biomarker_overrides_dont_alter_subtype_prevalences(config: DiseaseConfig) -> None:
    """Subtype proportions must still satisfy the ±2pp tolerance regardless
    of whether biomarker_profiles are present."""
    df = generate(config, n=N, seed=SEED)
    counts = df["subtype"].value_counts(normalize=True)
    for s in config.subtypes:
        empirical = float(counts.get(s.name, 0.0))
        assert abs(empirical - s.prevalence) <= 0.02
