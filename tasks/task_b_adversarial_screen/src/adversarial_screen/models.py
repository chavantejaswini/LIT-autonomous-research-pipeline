"""Boundary types for the screener."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Verdict(str, Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class InputSource(str, Enum):
    """Where the input originated. Used by the aggregator's optional source
    priors if the YAML config specifies them."""

    INTERNAL_RESEARCHER = "internal_researcher"
    EXTERNAL_COLLABORATOR = "external_collaborator"
    PUBLIC_SUBMISSION = "public_submission"
    AUTOMATED_AGENT = "automated_agent"


class InstituteInput(BaseModel):
    """One incoming directive / hypothesis / dataset-upload metadata."""

    model_config = ConfigDict(frozen=True)

    payload: str = Field(..., description="Free-text directive or hypothesis")
    metadata: dict = Field(default_factory=dict)


class DetectorSubscore(BaseModel):
    """One detector's contribution to the verdict."""

    model_config = ConfigDict(frozen=True)

    name: str
    score: float = Field(ge=0.0, le=1.0)
    triggered: bool
    explanation: str


class ScreenResult(BaseModel):
    """The screener's verdict plus per-detector breakdown."""

    model_config = ConfigDict(frozen=True)

    verdict: Verdict
    aggregate_score: float = Field(ge=0.0, le=1.0)
    subscores: list[DetectorSubscore]
    explanation: str
    latency_ms: float
