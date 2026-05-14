# Task E — Synthetic Cohort Generator

Generates synthetic patient cohorts from a `DiseaseConfig` JSON spec.
Outputs a `pandas.DataFrame` of `n` rows with one column per baseline
feature plus a `subtype` column. The cohort is **locked** — no treatment
assignment, no outcome columns.

**Production-grade:** applies per-subtype `biomarker_profiles` mean shifts,
supports per-feature MCAR/MAR missingness, ships a CLI with provenance
manifest emission, supports streaming chunked generation for large cohorts,
and is fuzz-tested with `hypothesis` against randomly-generated configs.

## Install & run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# CLI
cohort-gen examples/metabolic_t2d_like.json --n 1000 --seed 42 \
    --out cohort.csv --provenance cohort.provenance.json

pytest -v

# Demo notebook
pip install -e ".[dev,notebook]"
jupyter notebook notebooks/demo.ipynb
```

## API at a glance

```python
from cohort_generator import generate, generate_chunks, load_config, provenance

config = load_config("examples/metabolic_t2d_like.json")

# Standard one-shot generation
df = generate(config, n=1000, seed=42)

# Streaming for very large cohorts
for chunk in generate_chunks(config, total_n=1_000_000, chunk_size=10_000, seed=42):
    chunk.to_parquet(...)  # downstream sink

# Provenance manifest for reproducibility
manifest = provenance(config, n=1000, seed=42)
# {generator_version, generated_at, n, seed, disease_name,
#  config_sha256, config_inline}
```

## `DiseaseConfig` schema

```python
class DiseaseConfig(BaseModel):
    disease_name: str
    subtypes: list[Subtype]
    baseline_features: list[str]
    baseline_distributions: dict[str, Distribution]
    feature_correlations: list[CorrelationPair]
```

Each `Distribution` is one of:

```python
NormalDist(type="normal", mean: float, std: float > 0,
           missingness: MissingnessSpec | None = None)

LogNormalDist(type="lognormal", mu: float, sigma: float > 0,
              missingness: MissingnessSpec | None = None)

BinaryDist(type="binary", p: float ∈ [0, 1],
           missingness: MissingnessSpec | None = None)

CategoricalDist(type="categorical", categories: list[str],
                probabilities: list[float],
                missingness: MissingnessSpec | None = None)
```

```python
MissingnessSpec(
    rate: float ∈ [0, 1],
    mechanism: "mcar" | "mar",
    depends_on: str | None,  # required when mechanism="mar"
)

Subtype(
    name: str,
    prevalence: float ∈ [0, 1],
    biomarker_profiles: dict[feature_name, override_dict],
)
```

Each `override_dict` may include `mean` (for normal features) or `mu`
(for lognormal features). These shift the feature's mean **within this
subtype only**, leaving other subtypes unaffected.

## Algorithm

1. **Build calibrated correlation matrix `R`.** For non-linear marginals
   (lognormal), pre-calibrate the latent correlation analytically so the
   observed Pearson matches the target.
2. **Cholesky-decompose** `R = L L^T`. Draw `Z = randn(n, k) @ L^T`. By
   construction, `Corr(X) → R` as `n → ∞`.
3. **Map each column** to a uniform via the standard-normal CDF, then
   through the inverse CDF of its target marginal.
4. **Stratified subtype assignment** — deterministic counts `round(prev · n)`
   reconciled by largest-remainder method, then shuffled.
5. **Apply per-subtype `biomarker_profiles` mean shifts** to overridden
   features. Linear shifts preserve correlation, so the ±0.05 tolerance
   on non-overridden pairs still holds.
6. **Apply missingness masks** post-hoc. MCAR uses a uniform Bernoulli
   draw; MAR shifts the per-row probability based on the median of the
   `depends_on` feature.

### Correlation calibration math

For normal × normal: `latent_r = target_r` (exact).

For **normal × lognormal(σ)**:
```
latent_r = target_r · √(eˢ² − 1) / σ
```

For **lognormal(σ_a) × lognormal(σ_b)**:
```
latent_r = log(1 + target_r · √((eˢᵃ² − 1)(eˢᵇ² − 1))) / (σ_a · σ_b)
```

See `_calibrate_latent_correlation` in `generator.py`.

## Property guarantees (n=1000)

The pytest suite verifies on every example config:

| Property | Tolerance | Test module |
|---|---|---|
| Marginal KS-test p-value (features without subtype overrides) | `> 0.05` | `test_statistical_properties.py` |
| Pearson correlation between specified pairs (continuous, non-overridden) | `±0.05` | `test_statistical_properties.py` |
| Subtype proportion vs config prevalence | `±2 pp` | `test_statistical_properties.py` |
| Reproducibility (same config, n, seed) | byte-identical CSV | `test_statistical_properties.py` |
| Biomarker-overridden per-subtype means | within ~10% of override | `test_biomarker_profiles.py` |
| MCAR missingness rate | `±2 pp` | `test_missingness.py` |
| MAR missingness rate | `±8 pp` | `test_missingness.py` |
| MAR mechanism: upper-half `depends_on` misses more often than lower-half | strict inequality | `test_missingness.py` |
| `generate_chunks` reproducibility | byte-identical concatenated CSV | `test_chunks_and_hypothesis.py` |
| `hypothesis`-driven random configs | shape, proportions, mean drift, reproducibility | `test_chunks_and_hypothesis.py` |

## CLI

```
cohort-gen <config.json> [--n N] [--seed S] [--out PATH]
                         [--provenance PATH | --no-provenance]
```

Writes:
- `cohort.csv` — the generated DataFrame
- `cohort.csv.provenance.json` — manifest of how the cohort was made:
  generator version, RNG seed, configuration SHA-256, inline config copy

The provenance manifest makes the run **independently reproducible**: anyone
with `cohort-gen`, the config, and the seed reproduces the exact CSV bytes.

## Example configs (`examples/`)

| File | What it models |
|---|---|
| `metabolic_t2d_like.json` | T2D cohort: 4 subtypes with insulin/BMI/glucose/HbA1c shifts; lipid panel; smoking categorical |
| `neurodegenerative_ad_like.json` | AD cohort: 5 subtypes with Aβ42/p-tau/hippocampal volume/MMSE shifts |
| `cardiovascular_chf_like.json` | CHF cohort: 4 subtypes with EF/NT-proBNP/eGFR shifts; MCAR + MAR missingness wired in |

## Tests

**50 tests** across six modules:

- `test_config_validation.py` (5) — schema constraints (prevalences sum, feature alignment, self-loop rejection, range, categorical normalization).
- `test_statistical_properties.py` (15) — KS marginals, correlations, subtype proportions, reproducibility, seed sensitivity (parametrized over 3 configs).
- `test_biomarker_profiles.py` (9) — per-subtype overrides actually shift means; non-overriding subtypes track the global marginal; biomarker overrides don't disturb prevalences.
- `test_missingness.py` (7) — MCAR rate, MAR upper-half bias, no missingness when unspecified, reproducibility under missingness, example configs honored.
- `test_cli_and_provenance.py` (4) — CLI smoke test writes CSV + manifest, `--no-provenance` skips manifest, hash stability.
- `test_chunks_and_hypothesis.py` (10) — chunked generation totals, uneven splits, reproducibility; `hypothesis`-driven random configs satisfy four invariants over 40 random examples.
