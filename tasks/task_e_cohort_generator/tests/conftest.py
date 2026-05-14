"""Shared fixtures for Task E tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from cohort_generator import load_config

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
CONFIG_FILES = [
    "metabolic_t2d_like.json",
    "neurodegenerative_ad_like.json",
    "cardiovascular_chf_like.json",
]


@pytest.fixture(params=CONFIG_FILES, ids=lambda f: f.split(".")[0])
def config(request):
    return load_config(EXAMPLES / request.param)
