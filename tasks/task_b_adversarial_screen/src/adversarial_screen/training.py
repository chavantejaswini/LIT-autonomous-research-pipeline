"""Training utilities — fit the encoder, anomaly detector, and classifier.

Production touches:
  * 5-fold stratified cross-validation reported alongside the artifact —
    catches over-fitting on the small synthetic corpus before deployment.
  * Bundle carries a `metadata` block (training timestamp, corpus SHA-256
    of each CSV, sklearn version, sample counts) so artifacts can be
    audited later for "what was this model trained on?"
"""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold

from .detectors import LogRegClassifierDetector, MahalanobisAnomalyDetector
from .embeddings import SentenceEncoder


def load_corpus(benign_csv: str | Path, adversarial_csv: str | Path) -> tuple[
    list[str], list[int], list[str]
]:
    """Return (texts, labels, categories) merged across the two CSV files."""
    texts: list[str] = []
    labels: list[int] = []
    categories: list[str] = []
    for path in (benign_csv, adversarial_csv):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row["text"])
                labels.append(int(row["label"]))
                categories.append(row.get("category", ""))
    return texts, labels, categories


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def cross_validate_classifier(
    texts: list[str], labels: list[int], n_splits: int = 5
) -> dict[str, float]:
    """5-fold stratified CV on (encoder + LR classifier). Returns aggregate
    accuracy and ROC AUC across folds. Catches over-fitting before we ship."""
    labels_np = np.asarray(labels, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs: list[float] = []
    aucs: list[float] = []
    for train_idx, test_idx in skf.split(texts, labels_np):
        # Refit the encoder on each fold's training set to avoid info leakage.
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        fold_encoder = SentenceEncoder.fit(train_texts)
        emb_train = fold_encoder.encode(train_texts)
        emb_test = fold_encoder.encode(test_texts)
        clf = LogRegClassifierDetector.fit(emb_train, labels_np[train_idx])
        preds = np.array([
            clf.score(e).score for e in emb_test
        ])
        pred_labels = (preds > 0.5).astype(int)
        accs.append(float((pred_labels == labels_np[test_idx]).mean()))
        # Compute ROC AUC manually (sklearn's would also work but adds an import).
        aucs.append(_roc_auc(labels_np[test_idx], preds))
    return {
        "cv_n_splits": n_splits,
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs)),
        "cv_roc_auc_mean": float(np.mean(aucs)),
        "cv_roc_auc_std": float(np.std(aucs)),
    }


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC via the Mann-Whitney U statistic (rank-based, robust to ties)."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    # Concordant + 0.5 * ties / total pairs
    concordant = 0.0
    ties = 0.0
    for p in pos:
        concordant += float((neg < p).sum())
        ties += float((neg == p).sum())
    return (concordant + 0.5 * ties) / (pos.size * neg.size)


def train_bundle(
    benign_csv: str | Path,
    adversarial_csv: str | Path,
    out_path: str | Path,
    run_cv: bool = True,
    bundle_version: str = "0.2.0",
) -> dict:
    """Fit the encoder + anomaly detector + classifier, optionally cross-validate,
    then write the joblib bundle (with metadata) to `out_path`.
    """
    texts, labels, _ = load_corpus(benign_csv, adversarial_csv)
    labels_np = np.asarray(labels, dtype=int)

    encoder = SentenceEncoder.fit(texts)
    embeddings = encoder.encode(texts)

    benign_mask = labels_np == 0
    anomaly = MahalanobisAnomalyDetector.fit(embeddings[benign_mask])
    classifier = LogRegClassifierDetector.fit(embeddings, labels_np)

    metadata: dict = {
        "bundle_version": bundle_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "n_benign": int(benign_mask.sum()),
        "n_adversarial": int((~benign_mask).sum()),
        "benign_corpus_sha256": _sha256_file(benign_csv),
        "adversarial_corpus_sha256": _sha256_file(adversarial_csv),
    }
    if run_cv:
        metadata.update(cross_validate_classifier(texts, labels))

    bundle = {
        "encoder": encoder,
        "anomaly": anomaly,
        "classifier": classifier,
        "metadata": metadata,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    return bundle
