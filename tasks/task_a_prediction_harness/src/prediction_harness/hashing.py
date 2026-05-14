"""Content hashing for predictions.

A prediction's identity is the SHA-256 of its canonical JSON serialization
over (model_id, dataset_hash, prediction). Canonicalization (sorted keys,
no whitespace, ensure_ascii) means the same logical input always hashes to
the same digest regardless of dict ordering or pretty-printing.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(payload: Any) -> str:
    """Deterministic JSON serialization used as the basis for content hashing."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_default,
    )


def _default(obj: Any) -> Any:
    # SQLAlchemy/datetime values etc. fall back to str — but we only ever
    # hash user-supplied dicts of JSON primitives, so this is defensive.
    return str(obj)


def content_hash(model_id: str, dataset_hash: str, prediction: dict) -> str:
    """Return the hex SHA-256 of the canonical content envelope.

    The envelope is a JSON object with three keys to prevent collisions
    between, e.g., a `model_id` whose value matches another prediction's
    `dataset_hash`.
    """
    envelope = {
        "model_id": model_id,
        "dataset_hash": dataset_hash,
        "prediction": prediction,
    }
    return hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()
