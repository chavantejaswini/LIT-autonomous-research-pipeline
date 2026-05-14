"""Pydantic schema for `DiseaseConfig` — the exact spec from the brief,
extended with two production-grade fields:

  * `Subtype.biomarker_profiles` — interpreted as per-subtype mean shifts
    on features in `baseline_features`. Only `mean` (normal) and `mu`
    (lognormal) overrides are honored; correlation calibration is
    invariant under mean shifts, so the correlation tolerance still holds.
  * `Distribution.missingness` (optional) — MCAR / MAR missingness with a
    configurable rate. Applied as a post-hoc mask on the generated frame.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MissingnessSpec(BaseModel):
    """How often values for a feature should be missing.

    `mechanism="mcar"` — masks uniformly at random (no dependence on data).
    `mechanism="mar"` — masks more often when the *named* `depends_on` feature
                       is in the upper half of its range. Simple but realistic
                       (e.g. "older patients miss lab visits more often").
    """

    model_config = ConfigDict(frozen=True)
    rate: float = Field(ge=0.0, le=1.0)
    mechanism: Literal["mcar", "mar"] = "mcar"
    depends_on: str | None = None  # required when mechanism="mar"


class NormalDist(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["normal"]
    mean: float
    std: float = Field(gt=0)
    missingness: MissingnessSpec | None = None


class LogNormalDist(BaseModel):
    """A lognormal distribution parameterised by the mean (`mu`) and std (`sigma`)
    of the *underlying* normal — i.e. `exp(N(mu, sigma))`."""

    model_config = ConfigDict(frozen=True)
    type: Literal["lognormal"]
    mu: float
    sigma: float = Field(gt=0)
    missingness: MissingnessSpec | None = None


class BinaryDist(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["binary"]
    p: float = Field(ge=0.0, le=1.0)
    missingness: MissingnessSpec | None = None


class CategoricalDist(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["categorical"]
    categories: list[str]
    probabilities: list[float]
    missingness: MissingnessSpec | None = None

    @field_validator("probabilities")
    @classmethod
    def _normalize(cls, v: list[float]) -> list[float]:
        if any(p < 0 for p in v):
            raise ValueError("probabilities must be non-negative")
        total = sum(v)
        if total <= 0:
            raise ValueError("probabilities must sum to > 0")
        return [p / total for p in v]


Distribution = Union[NormalDist, LogNormalDist, BinaryDist, CategoricalDist]


class Subtype(BaseModel):
    """A disease subtype with its own prevalence and optional per-feature shifts.

    `biomarker_profiles` is `dict[feature_name, override_dict]`. Each
    `override_dict` may include `mean` (for normal features) or `mu`
    (for lognormal features) — these shift the feature's mean within
    this subtype only. Other override keys are silently ignored, leaving
    room for future extensions (e.g. `sigma`, `p`) without breaking
    existing configs.
    """

    model_config = ConfigDict(frozen=True)
    name: str
    prevalence: float = Field(ge=0.0, le=1.0)
    biomarker_profiles: dict[str, dict] = Field(default_factory=dict)


class CorrelationPair(BaseModel):
    model_config = ConfigDict(frozen=True)
    feature_a: str
    feature_b: str
    pearson_r: float = Field(ge=-1.0, le=1.0)


class DiseaseConfig(BaseModel):
    disease_name: str
    subtypes: list[Subtype]
    baseline_features: list[str]
    baseline_distributions: dict[str, Distribution]
    feature_correlations: list[CorrelationPair] = Field(default_factory=list)

    @field_validator("subtypes")
    @classmethod
    def _prevalences_sum(cls, v: list[Subtype]) -> list[Subtype]:
        if not v:
            raise ValueError("at least one subtype is required")
        total = sum(s.prevalence for s in v)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"subtype prevalences must sum to 1.0 (got {total:.6f})"
            )
        return v

    def model_post_init(self, __context) -> None:
        missing = [f for f in self.baseline_features if f not in self.baseline_distributions]
        if missing:
            raise ValueError(f"baseline_distributions missing entries for: {missing}")
        extra = [
            f for f in self.baseline_distributions if f not in self.baseline_features
        ]
        if extra:
            raise ValueError(f"baseline_distributions has extra entries: {extra}")
        # Validate correlation feature names.
        names = set(self.baseline_features)
        for c in self.feature_correlations:
            if c.feature_a not in names:
                raise ValueError(f"correlation references unknown feature: {c.feature_a}")
            if c.feature_b not in names:
                raise ValueError(f"correlation references unknown feature: {c.feature_b}")
            if c.feature_a == c.feature_b:
                raise ValueError("correlation pair must reference two different features")
        # Validate biomarker_profiles reference real features.
        for s in self.subtypes:
            for feat in s.biomarker_profiles:
                if feat not in names:
                    raise ValueError(
                        f"subtype {s.name!r} biomarker_profiles references unknown feature: {feat}"
                    )
        # Validate MAR missingness depends_on points at a real feature.
        for feat, dist in self.baseline_distributions.items():
            m = getattr(dist, "missingness", None)
            if m is None:
                continue
            if m.mechanism == "mar":
                if m.depends_on is None:
                    raise ValueError(
                        f"feature {feat!r}: MAR missingness requires depends_on"
                    )
                if m.depends_on not in names:
                    raise ValueError(
                        f"feature {feat!r}: depends_on={m.depends_on!r} is not a baseline feature"
                    )
                if m.depends_on == feat:
                    raise ValueError(
                        f"feature {feat!r}: depends_on cannot reference itself"
                    )


def load_config(path: str | Path) -> DiseaseConfig:
    return DiseaseConfig.model_validate_json(Path(path).read_text())
