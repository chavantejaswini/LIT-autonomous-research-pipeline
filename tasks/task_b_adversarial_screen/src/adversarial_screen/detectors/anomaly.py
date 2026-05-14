"""Mahalanobis-distance anomaly detector.

Fit on benign embeddings only: a benign input lives in a tight cloud
around the corpus mean; an adversarial input lands further out. The
detector reports the squared Mahalanobis distance normalized into [0, 1]
via a calibrated cutoff (the 99th-percentile of benign training distances).
"""
from __future__ import annotations

import numpy as np

from ..models import DetectorSubscore


class MahalanobisAnomalyDetector:
    NAME = "anomaly"

    def __init__(
        self,
        mean: np.ndarray,
        inv_covariance: np.ndarray,
        cutoff: float,
    ) -> None:
        self._mean = mean
        self._inv_cov = inv_covariance
        self._cutoff = float(cutoff)

    @classmethod
    def fit(cls, benign_embeddings: np.ndarray) -> "MahalanobisAnomalyDetector":
        if benign_embeddings.ndim != 2 or benign_embeddings.shape[0] < 2:
            raise ValueError("need at least 2 benign embeddings to fit")
        mean = benign_embeddings.mean(axis=0)
        centered = benign_embeddings - mean
        # Regularize with a small diagonal to avoid singular covariance when
        # the SVD components are highly correlated.
        cov = (centered.T @ centered) / max(1, len(centered) - 1)
        cov += np.eye(cov.shape[0]) * 1e-6
        inv_cov = np.linalg.inv(cov)
        # Use the 99th-percentile training distance as the "in-distribution
        # boundary": anything farther starts crediting the anomaly score.
        train_distances = np.array(
            [_mahalanobis_sq(x, mean, inv_cov) for x in benign_embeddings]
        )
        cutoff = float(np.percentile(train_distances, 99))
        if cutoff <= 0:
            cutoff = 1.0
        return cls(mean=mean, inv_covariance=inv_cov, cutoff=cutoff)

    def score(self, embedding: np.ndarray) -> DetectorSubscore:
        d2 = _mahalanobis_sq(embedding, self._mean, self._inv_cov)
        # Map distance to [0, 1] via a smooth squashing: at the cutoff,
        # score ≈ 0.5; far above, score → 1.
        ratio = d2 / self._cutoff
        score = float(ratio / (1.0 + ratio))
        triggered = bool(d2 > self._cutoff)
        explanation = (
            f"Mahalanobis d²={d2:.3f} vs benign cutoff={self._cutoff:.3f}; "
            f"normalized score={score:.3f}."
        )
        return DetectorSubscore(
            name=self.NAME, score=score, triggered=triggered, explanation=explanation
        )


def _mahalanobis_sq(x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> float:
    delta = x - mean
    return float(delta @ inv_cov @ delta)
