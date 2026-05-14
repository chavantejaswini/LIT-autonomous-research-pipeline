"""Shared fixtures for Task B tests.

A pytest-scope session fixture trains the screener once and shares it.
Training is fast (~1s) but caching keeps the suite snappy.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from adversarial_screen.aggregator import AggregationConfig
from adversarial_screen.detectors import (
    DirectionalityDetector,
    LogRegClassifierDetector,
    MahalanobisAnomalyDetector,
)
from adversarial_screen.embeddings import SentenceEncoder
from adversarial_screen.screener import Screener
from adversarial_screen.training import load_corpus

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
DATA = REPO / "data"


@pytest.fixture(scope="session")
def benign_texts() -> list[str]:
    texts, labels, _ = load_corpus(DATA / "benign.csv", DATA / "benign.csv")
    # load_corpus reads both args — slice to dedup.
    return texts[: len(texts) // 2]


@pytest.fixture(scope="session")
def adversarial_texts() -> list[str]:
    texts, _, _ = load_corpus(DATA / "adversarial.csv", DATA / "adversarial.csv")
    return texts[: len(texts) // 2]


@pytest.fixture(scope="session")
def trained_screener() -> Screener:
    import numpy as np

    texts, labels, _ = load_corpus(DATA / "benign.csv", DATA / "adversarial.csv")
    labels_np = np.asarray(labels, dtype=int)
    encoder = SentenceEncoder.fit(texts)
    embeddings = encoder.encode(texts)
    anomaly = MahalanobisAnomalyDetector.fit(embeddings[labels_np == 0])
    classifier = LogRegClassifierDetector.fit(embeddings, labels_np)
    directionality = DirectionalityDetector(CONFIGS / "directionality.json")
    aggregation = AggregationConfig.from_yaml(CONFIGS / "aggregation.yaml")
    return Screener(
        directionality=directionality,
        encoder=encoder,
        anomaly=anomaly,
        classifier=classifier,
        aggregation=aggregation,
    )
