"""A lightweight sentence-embedding pipeline.

We don't pull in a transformer model: at the perimeter we need single-digit
millisecond latency per input. A TF-IDF vectorizer (word + char n-grams)
followed by `TruncatedSVD` to a fixed-dimensional latent space gives us a
stable, deterministic, sub-millisecond embedding that works well enough
for both the Mahalanobis anomaly detector and the LR classifier.

This same encoder is fit once on the benign corpus during training and
reused by every detector that needs a vector representation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


@dataclass
class SentenceEncoder:
    """Wraps a `TfidfVectorizer -> TruncatedSVD` pipeline.

    The pipeline is exposed as `.pipeline` so it can be pickled with joblib.
    """

    pipeline: Pipeline

    @classmethod
    def fit(cls, texts: list[str], n_components: int = 64) -> "SentenceEncoder":
        # SVD components must be < min(n_samples, n_features). Clamp to the
        # corpus size minus one so small corpora still train cleanly.
        n_components = min(n_components, max(2, len(texts) - 1))
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=0.95,
                        sublinear_tf=True,
                    ),
                ),
                ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
            ]
        )
        pipeline.fit(texts)
        return cls(pipeline=pipeline)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.pipeline.transform(texts))

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
