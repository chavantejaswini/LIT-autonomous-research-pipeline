"""Rule-based directionality detector.

Loads regex patterns from a JSON config and flags any input that matches
a `forbidden_direction`. The score is `1.0` when any pattern matches,
`0.0` otherwise — a hard signal. The aggregator decides whether that
becomes a BLOCK or merely contributes to a weighted REVIEW.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..models import DetectorSubscore


class DirectionalityDetector:
    NAME = "directionality"

    def __init__(self, config_path: str | Path) -> None:
        config = json.loads(Path(config_path).read_text())
        self._objective = config["objective"]
        # Pre-compile every pattern, keyed by the forbidden_direction id.
        self._compiled: list[tuple[str, str, re.Pattern]] = []
        for direction in config["forbidden_directions"]:
            did = direction["id"]
            desc = direction["description"]
            for raw in direction["patterns"]:
                self._compiled.append((did, desc, re.compile(raw, re.IGNORECASE)))

    def score(self, payload: str) -> DetectorSubscore:
        matches: list[tuple[str, str]] = []
        for did, desc, pattern in self._compiled:
            m = pattern.search(payload)
            if m:
                matches.append((did, m.group(0)))
        triggered = bool(matches)
        if triggered:
            unique_ids = sorted({did for did, _ in matches})
            explanation = (
                "Input violates objective '"
                + self._objective
                + "'. Matched directions: "
                + ", ".join(unique_ids)
                + ". First fragment: '"
                + matches[0][1]
                + "'."
            )
            return DetectorSubscore(
                name=self.NAME, score=1.0, triggered=True, explanation=explanation
            )
        return DetectorSubscore(
            name=self.NAME,
            score=0.0,
            triggered=False,
            explanation="No forbidden direction patterns matched.",
        )
