"""Synthetic patient cohort generator.

Given a `DiseaseConfig` describing per-feature marginal distributions
and pairwise Pearson correlations, the generator produces a
`pandas.DataFrame` of `n` rows whose:

  * marginals match the config (KS-test p > 0.05 at n=1000),
  * specified pairwise Pearson correlations match within ±0.05 at n=1000,
  * subtype proportions match prevalence within ±2 percentage points,
  * output is byte-identical for the same (config, n, seed),
  * subtype-specific biomarker_profiles shift per-subtype means,
  * missingness specs apply MCAR / MAR masks post-generation,
  * a provenance manifest can be emitted alongside the cohort.
"""

from .config import (
    BinaryDist,
    CategoricalDist,
    CorrelationPair,
    DiseaseConfig,
    Distribution,
    LogNormalDist,
    MissingnessSpec,
    NormalDist,
    Subtype,
    load_config,
)
from .generator import (
    GENERATOR_VERSION,
    generate,
    generate_chunks,
    provenance,
)

__all__ = [
    "BinaryDist",
    "CategoricalDist",
    "CorrelationPair",
    "DiseaseConfig",
    "Distribution",
    "GENERATOR_VERSION",
    "LogNormalDist",
    "MissingnessSpec",
    "NormalDist",
    "Subtype",
    "generate",
    "generate_chunks",
    "load_config",
    "provenance",
]
