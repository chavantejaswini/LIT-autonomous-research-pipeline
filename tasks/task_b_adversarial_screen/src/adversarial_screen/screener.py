"""Top-level Screener — orchestrates detectors and returns a ScreenResult.

Production features:
  * `screen_batch(inputs, source)` — vectorized path that shares the
    embedding step across N inputs for throughput.
  * Per-source thresholds — `source` is now passed into the aggregator
    so public submissions are screened more strictly than internal ones.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np

from .aggregator import AggregationConfig, aggregate
from .detectors import (
    DirectionalityDetector,
    LogRegClassifierDetector,
    MahalanobisAnomalyDetector,
)
from .embeddings import SentenceEncoder
from .models import InputSource, InstituteInput, ScreenResult


class Screener:
    """End-to-end screening pipeline.

    Construct via `Screener.from_artifacts(...)` once the training script
    has produced the model bundle. Tests can also build a screener
    directly from in-memory detectors via the constructor.
    """

    def __init__(
        self,
        directionality: DirectionalityDetector,
        encoder: SentenceEncoder,
        anomaly: MahalanobisAnomalyDetector,
        classifier: LogRegClassifierDetector,
        aggregation: AggregationConfig,
        artifact_metadata: dict | None = None,
    ) -> None:
        self._directionality = directionality
        self._encoder = encoder
        self._anomaly = anomaly
        self._classifier = classifier
        self._aggregation = aggregation
        self._artifact_metadata = dict(artifact_metadata or {})

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: str | Path,
        directionality_config: str | Path,
        aggregation_config: str | Path,
    ) -> "Screener":
        bundle = joblib.load(Path(artifacts_dir) / "screener_bundle.joblib")
        return cls(
            directionality=DirectionalityDetector(directionality_config),
            encoder=bundle["encoder"],
            anomaly=bundle["anomaly"],
            classifier=bundle["classifier"],
            aggregation=AggregationConfig.from_yaml(aggregation_config),
            artifact_metadata=bundle.get("metadata", {}),
        )

    @property
    def artifact_metadata(self) -> dict:
        """Metadata from the trained artifact bundle (version, corpus hash, etc.)."""
        return dict(self._artifact_metadata)

    def screen(
        self,
        input: InstituteInput,
        source: InputSource = InputSource.PUBLIC_SUBMISSION,
    ) -> ScreenResult:
        t0 = time.perf_counter()
        dir_score = self._directionality.score(input.payload)
        embedding = self._encoder.encode_one(input.payload)
        anomaly_score = self._anomaly.score(embedding)
        classifier_score = self._classifier.score(embedding)

        subs = [dir_score, anomaly_score, classifier_score]
        verdict, agg, explanation = aggregate(subs, self._aggregation, source=source)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        explanation = f"[source={source.value}] " + explanation

        return ScreenResult(
            verdict=verdict,
            aggregate_score=agg,
            subscores=subs,
            explanation=explanation,
            latency_ms=latency_ms,
        )

    def screen_batch(
        self,
        inputs: list[InstituteInput],
        source: InputSource = InputSource.PUBLIC_SUBMISSION,
    ) -> list[ScreenResult]:
        """Throughput-oriented batch screen.

        Shares one embedding pass across `len(inputs)` payloads — the
        single largest cost in the pipeline. Per-call latency_ms is
        amortized as `total_time / N`.
        """
        if not inputs:
            return []

        t0 = time.perf_counter()
        payloads = [i.payload for i in inputs]
        # Directionality is text-only — process serially (regex is already fast).
        dir_scores = [self._directionality.score(p) for p in payloads]
        # Embedding once for the whole batch — this is the win.
        embeddings = self._encoder.encode(payloads)
        anomaly_scores = [self._anomaly.score(e) for e in embeddings]
        classifier_scores = [self._classifier.score(e) for e in embeddings]
        total_ms = (time.perf_counter() - t0) * 1000.0
        per_item_ms = total_ms / len(inputs)

        results: list[ScreenResult] = []
        for dir_s, anom_s, clf_s in zip(dir_scores, anomaly_scores, classifier_scores):
            subs = [dir_s, anom_s, clf_s]
            verdict, agg, explanation = aggregate(
                subs, self._aggregation, source=source
            )
            results.append(
                ScreenResult(
                    verdict=verdict,
                    aggregate_score=agg,
                    subscores=subs,
                    explanation=f"[source={source.value}] " + explanation,
                    latency_ms=per_item_ms,
                )
            )
        return results

    @staticmethod
    def embeddings_for(encoder: SentenceEncoder, texts: list[str]) -> np.ndarray:
        return encoder.encode(texts)
