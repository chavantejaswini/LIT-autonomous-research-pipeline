"""Gaussian-copula generator.

Pipeline:
  1. Build target correlation matrix `R`, analytically pre-calibrated so
     the *observed* Pearson on the output matches the user's targets.
  2. Cholesky: `R = L L^T`. Draw `Z = randn(n, k) @ L^T`.
  3. Map each column through `Phi` to a uniform, then through the inverse
     CDF of the target marginal.
  4. Apply per-subtype mean shifts from `biomarker_profiles`.
  5. Apply post-hoc missingness masks (MCAR or MAR).
  6. Draw subtype assignments from a stratified sample so the proportions
     are exact (within rounding).

All randomness flows from a single `np.random.Generator(seed)` in a
fixed call order — output is byte-identical for any given (config, n, seed).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import (
    BinaryDist,
    CategoricalDist,
    DiseaseConfig,
    LogNormalDist,
    MissingnessSpec,
    NormalDist,
)

GENERATOR_VERSION = "0.2.0"


def generate(config: DiseaseConfig, n: int, seed: int) -> pd.DataFrame:
    """Produce an n-row cohort DataFrame for the given config and seed."""
    if n < 1:
        raise ValueError("n must be ≥ 1")
    rng = np.random.default_rng(seed)

    features = list(config.baseline_features)
    k = len(features)

    R = _build_calibrated_correlation_matrix(
        features, config.feature_correlations, config.baseline_distributions
    )
    R = _nearest_psd(R)
    L = np.linalg.cholesky(R)

    # Step 1: latent multivariate normal with correlation R.
    Z = rng.standard_normal(size=(n, k))
    X = Z @ L.T

    # Step 2: per-column marginal transform (global params).
    data: dict[str, np.ndarray] = {}
    for j, name in enumerate(features):
        u = norm.cdf(X[:, j])
        dist = config.baseline_distributions[name]
        data[name] = _apply_marginal(u, dist)

    # Step 3: stratified subtype assignment.
    data["subtype"] = _stratified_subtype_assignment(n, config, rng)

    # Step 4: apply subtype-specific mean shifts from biomarker_profiles.
    _apply_biomarker_overrides(data, config)

    # Step 5: apply missingness masks. Order matters — missingness reads
    # the (pre-mask) values of `depends_on` for MAR.
    _apply_missingness(data, config, rng)

    df = pd.DataFrame(data, columns=features + ["subtype"])
    return df


def generate_chunks(
    config: DiseaseConfig, total_n: int, chunk_size: int, seed: int
) -> Iterator[pd.DataFrame]:
    """Yield successive chunks of a large cohort.

    The seeded RNG produces deterministic chunks: concatenating yields
    yields the same rows in the same order. Useful for cohorts that
    exceed memory or stream into downstream pipelines.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be ≥ 1")
    rng_seed = seed
    emitted = 0
    while emitted < total_n:
        batch = min(chunk_size, total_n - emitted)
        # Derive a per-chunk seed deterministically so the chunked path
        # produces stable per-chunk data without sharing global RNG state.
        chunk_seed = _derive_chunk_seed(rng_seed, emitted)
        yield generate(config, n=batch, seed=chunk_seed)
        emitted += batch


def provenance(config: DiseaseConfig, n: int, seed: int) -> dict:
    """Return a sidecar manifest describing exactly how a cohort was made."""
    config_json = config.model_dump_json()
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    return {
        "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n": n,
        "seed": seed,
        "disease_name": config.disease_name,
        "config_sha256": config_hash,
        "config_inline": json.loads(config_json),
    }


# ---- subtype assignment ---------------------------------------------------


def _stratified_subtype_assignment(
    n: int, config: DiseaseConfig, rng: np.random.Generator
) -> np.ndarray:
    counts = [int(round(s.prevalence * n)) for s in config.subtypes]
    diff = n - sum(counts)
    if diff != 0:
        fractions = [
            (s.prevalence * n - int(round(s.prevalence * n)), i)
            for i, s in enumerate(config.subtypes)
        ]
        fractions.sort(reverse=(diff > 0))
        for k in range(abs(diff)):
            counts[fractions[k % len(fractions)][1]] += 1 if diff > 0 else -1
    assignment = np.empty(n, dtype=object)
    pos = 0
    for s, c in zip(config.subtypes, counts):
        assignment[pos : pos + c] = s.name
        pos += c
    rng.shuffle(assignment)
    return assignment


# ---- biomarker overrides --------------------------------------------------


def _apply_biomarker_overrides(data: dict[str, np.ndarray], config: DiseaseConfig) -> None:
    """Apply per-subtype mean shifts.

    For a normal feature: shift = override_mean − global_mean (added to rows).
    For a lognormal feature: shift on the *log* scale, then re-exponentiate.
    Binary / categorical features are not affected (no shift semantics).
    """
    if not any(s.biomarker_profiles for s in config.subtypes):
        return
    subtype_col = data["subtype"]
    for s in config.subtypes:
        if not s.biomarker_profiles:
            continue
        mask = subtype_col == s.name
        if not mask.any():
            continue
        for feat, override in s.biomarker_profiles.items():
            dist = config.baseline_distributions[feat]
            if isinstance(dist, NormalDist) and "mean" in override:
                shift = float(override["mean"]) - dist.mean
                data[feat][mask] = data[feat][mask] + shift
            elif isinstance(dist, LogNormalDist) and "mu" in override:
                shift = float(override["mu"]) - dist.mu
                # data is lognormal; shift on the log scale.
                logged = np.log(data[feat][mask])
                data[feat][mask] = np.exp(logged + shift)


# ---- missingness ----------------------------------------------------------


def _apply_missingness(
    data: dict[str, np.ndarray], config: DiseaseConfig, rng: np.random.Generator
) -> None:
    """Replace a fraction of values with NaN according to each feature's spec."""
    n = len(data["subtype"])
    for feat in config.baseline_features:
        dist = config.baseline_distributions[feat]
        spec: MissingnessSpec | None = getattr(dist, "missingness", None)
        if spec is None or spec.rate <= 0.0:
            continue

        if spec.mechanism == "mcar":
            mask = rng.random(n) < spec.rate
        elif spec.mechanism == "mar":
            # Higher missing rate when `depends_on` is in the upper half of its range.
            dep_vals = _to_float_array(data[spec.depends_on])
            median = np.nanmedian(dep_vals)
            # Bias the per-row missing probability up/down by ±50% of the rate.
            base = spec.rate
            high_p = min(1.0, base * 1.5)
            low_p = max(0.0, base * 0.5)
            per_row_p = np.where(dep_vals > median, high_p, low_p)
            mask = rng.random(n) < per_row_p
        else:
            continue  # pragma: no cover

        # Cast object arrays (binary 0/1 stored as int, or categorical strings)
        # need to switch to a NaN-capable dtype before masking.
        arr = data[feat]
        if arr.dtype != object and not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(float)
        if np.issubdtype(arr.dtype, np.floating):
            arr[mask] = np.nan
        else:
            arr[mask] = None  # object dtype — None reads as NaN in pandas
        data[feat] = arr


def _to_float_array(arr: np.ndarray) -> np.ndarray:
    """Coerce an object/binary array into floats so MAR thresholding works."""
    if arr.dtype == object:
        try:
            return arr.astype(float)
        except (TypeError, ValueError):
            # Categorical (string) features can't be thresholded directly;
            # fall back to a flat 0 array → MAR degenerates to MCAR.
            return np.zeros(len(arr))
    return arr.astype(float)


# ---- correlation matrix calibration ---------------------------------------


def _build_calibrated_correlation_matrix(
    features: list[str], pairs, distributions: dict,
) -> np.ndarray:
    idx = {f: i for i, f in enumerate(features)}
    k = len(features)
    R = np.eye(k)
    for c in pairs:
        dist_a = distributions[c.feature_a]
        dist_b = distributions[c.feature_b]
        latent = _calibrate_latent_correlation(c.pearson_r, dist_a, dist_b)
        i, j = idx[c.feature_a], idx[c.feature_b]
        R[i, j] = latent
        R[j, i] = latent
    return R


def _calibrate_latent_correlation(target: float, dist_a, dist_b) -> float:
    """Invert the marginal-transform's effect on Pearson correlation."""
    a_norm = isinstance(dist_a, NormalDist)
    b_norm = isinstance(dist_b, NormalDist)
    a_log = isinstance(dist_a, LogNormalDist)
    b_log = isinstance(dist_b, LogNormalDist)

    if a_norm and b_norm:
        return float(target)
    if a_norm and b_log:
        s = dist_b.sigma
        factor = s / np.sqrt(np.exp(s * s) - 1)
        return float(np.clip(target / factor, -0.999, 0.999))
    if a_log and b_norm:
        s = dist_a.sigma
        factor = s / np.sqrt(np.exp(s * s) - 1)
        return float(np.clip(target / factor, -0.999, 0.999))
    if a_log and b_log:
        sa, sb = dist_a.sigma, dist_b.sigma
        denom = np.sqrt((np.exp(sa * sa) - 1) * (np.exp(sb * sb) - 1))
        inside = 1.0 + target * denom
        if inside <= 0:
            return -0.999 if target < 0 else 0.999
        return float(np.clip(np.log(inside) / (sa * sb), -0.999, 0.999))
    # Discrete/mixed cases — best-effort fallback.
    return float(target)


def _nearest_psd(R: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    R = (R + R.T) / 2
    w, V = np.linalg.eigh(R)
    w = np.clip(w, eps, None)
    A = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(A))
    A = A / np.outer(d, d)
    A = (A + A.T) / 2
    return A


# ---- marginal inverse CDFs ------------------------------------------------


def _apply_marginal(u: np.ndarray, dist) -> np.ndarray:
    """Map a uniform `u` ∈ [0, 1] through the inverse CDF of `dist`."""
    eps = np.finfo(float).eps
    u = np.clip(u, eps, 1 - eps)

    if isinstance(dist, NormalDist):
        return norm.ppf(u, loc=dist.mean, scale=dist.std)
    if isinstance(dist, LogNormalDist):
        return np.exp(norm.ppf(u, loc=dist.mu, scale=dist.sigma))
    if isinstance(dist, BinaryDist):
        return (u < dist.p).astype(int)
    if isinstance(dist, CategoricalDist):
        cum = np.cumsum(dist.probabilities)
        idx = np.searchsorted(cum, u, side="right")
        idx = np.clip(idx, 0, len(dist.categories) - 1)
        return np.array([dist.categories[i] for i in idx], dtype=object)
    raise TypeError(f"unknown distribution type: {type(dist).__name__}")


def _derive_chunk_seed(base_seed: int, offset: int) -> int:
    """Deterministic chunk seed — keeps `generate_chunks` reproducible."""
    h = hashlib.sha256(f"{base_seed}:{offset}".encode()).hexdigest()
    return int(h[:8], 16)
