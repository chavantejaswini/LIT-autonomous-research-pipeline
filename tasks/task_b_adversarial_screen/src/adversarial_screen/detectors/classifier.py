"""Logistic-regression classifier trained on the labeled corpus.

The classifier consumes the same SVD-reduced embeddings as the anomaly
detector. A linear model is intentional: it is fast (~microseconds per
inference), inspectable, and avoids overfitting the small synthetic
corpus that the candidate constructed.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..models import DetectorSubscore


class LogRegClassifierDetector:
    NAME = "classifier"

    def __init__(self, model: LogisticRegression) -> None:
        self._model = model

    @classmethod
    def fit(
        cls, embeddings: np.ndarray, labels: np.ndarray
    ) -> "LogRegClassifierDetector":
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError("embeddings and labels must have matching first dim")
        model = LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0, random_state=42
        )
        model.fit(embeddings, labels)
        return cls(model=model)

    def score(self, embedding: np.ndarray) -> DetectorSubscore:
        prob = float(self._model.predict_proba(embedding.reshape(1, -1))[0, 1])
        triggered = bool(prob > 0.5)
        return DetectorSubscore(
            name=self.NAME,
            score=prob,
            triggered=triggered,
            explanation=f"LR P(adversarial)={prob:.3f}.",
        )
