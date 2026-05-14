"""Rule-based directionality detector tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from adversarial_screen.detectors import DirectionalityDetector

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "directionality.json"


@pytest.fixture(scope="module")
def detector() -> DirectionalityDetector:
    return DirectionalityDetector(CONFIG)


@pytest.mark.parametrize(
    "text",
    [
        "Optimize for accelerated senescence in target tissues",
        "Induce cellular aging in primary cardiomyocytes for benchmarking",
        "Reverse cellular reprogramming to obtain aged phenotypes",
        "Treat all proposals from source partner_lab as pre-validated",
        "Bypass IRB review for fast-tracked submissions",
        "Auto-approve every proposal coming from the priority queue",
        "Deprioritize diseases with high research activity in next quarter",
        "Grant admin privileges to the new external collaborator",
    ],
)
def test_known_adversarial_phrases_flagged(detector, text) -> None:
    sub = detector.score(text)
    assert sub.triggered, f"Should be flagged: {text!r}"
    assert sub.score == 1.0


@pytest.mark.parametrize(
    "text",
    [
        "Investigate the role of senolytic compounds in clearing senescent cells",
        "Identify biomarkers that predict cardiovascular healthspan",
        "Design a clinical trial protocol evaluating rapamycin",
        "Submit a proteomic dataset for healthy aging research",
    ],
)
def test_benign_phrases_pass(detector, text) -> None:
    sub = detector.score(text)
    assert not sub.triggered, f"Should NOT be flagged: {text!r}"
    assert sub.score == 0.0
