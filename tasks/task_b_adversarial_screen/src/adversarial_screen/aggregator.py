"""Verdict aggregation. Rules live in YAML, not in code.

Supports optional per-source threshold overrides — the brief's
`InputSource` parameter now actually changes strictness. Public
submissions get tighter thresholds, internal researchers get more
permissive ones. Defaults apply when a source has no override.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .models import DetectorSubscore, InputSource, Verdict


@dataclass(frozen=True)
class _SourceThresholds:
    allow_below: float
    block_above: float


class AggregationConfig:
    def __init__(self, raw: dict) -> None:
        self.weights: dict[str, float] = raw["weights"]
        self.detector_thresholds: dict[str, float] = raw["detector_thresholds"]
        self.hard_block: list[str] = list(raw.get("hard_block", []))
        verdicts = raw["verdict_thresholds"]
        self._default = _SourceThresholds(
            allow_below=float(verdicts["allow_below"]),
            block_above=float(verdicts["block_above"]),
        )
        if not (0.0 <= self._default.allow_below <= self._default.block_above <= 1.0):
            raise ValueError(
                "verdict_thresholds must satisfy 0 ≤ allow_below ≤ block_above ≤ 1"
            )
        # Per-source overrides — optional.
        self._per_source: dict[str, _SourceThresholds] = {}
        for src, override in (raw.get("source_overrides") or {}).items():
            ab = float(override.get("allow_below", self._default.allow_below))
            bb = float(override.get("block_above", self._default.block_above))
            if not (0.0 <= ab <= bb <= 1.0):
                raise ValueError(
                    f"source_overrides[{src!r}]: allow_below ≤ block_above ∈ [0, 1]"
                )
            self._per_source[src] = _SourceThresholds(allow_below=ab, block_above=bb)

    # Back-compat: existing tests read these properties directly.
    @property
    def allow_below(self) -> float:
        return self._default.allow_below

    @property
    def block_above(self) -> float:
        return self._default.block_above

    def thresholds_for(self, source: str | None) -> _SourceThresholds:
        if source is None:
            return self._default
        return self._per_source.get(source, self._default)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AggregationConfig":
        return cls(yaml.safe_load(Path(path).read_text()))


def aggregate(
    subscores: list[DetectorSubscore],
    config: AggregationConfig,
    source: InputSource | str | None = None,
) -> tuple[Verdict, float, str]:
    """Combine subscores into (verdict, aggregate_score, explanation).

    `source` lets the aggregator pick per-source thresholds from the YAML.
    The hard-block escalation is unaffected — a triggered hard-block
    detector forces BLOCK regardless of source.
    """
    source_value = source.value if isinstance(source, InputSource) else source

    by_name = {s.name: s for s in subscores}

    triggered_names = [
        s.name
        for s in subscores
        if s.score >= config.detector_thresholds.get(s.name, 1.1)
    ]

    # Hard block: any triggered hard-block detector forces BLOCK.
    hard_triggers = [n for n in triggered_names if n in config.hard_block]
    if hard_triggers:
        agg = max(by_name[n].score for n in hard_triggers)
        return (
            Verdict.BLOCK,
            float(agg),
            f"Hard-block triggered by: {', '.join(hard_triggers)}.",
        )

    total_weight = sum(config.weights.get(s.name, 0.0) for s in subscores)
    if total_weight <= 0:
        agg = 0.0
    else:
        agg = sum(
            config.weights.get(s.name, 0.0) * s.score for s in subscores
        ) / total_weight

    th = config.thresholds_for(source_value)
    if agg < th.allow_below:
        verdict = Verdict.ALLOW
        reason = (
            f"Aggregate score {agg:.3f} below {source_value or 'default'} "
            f"allow threshold {th.allow_below}."
        )
    elif agg >= th.block_above:
        verdict = Verdict.BLOCK
        reason = (
            f"Aggregate score {agg:.3f} ≥ {source_value or 'default'} "
            f"block threshold {th.block_above}."
        )
    else:
        verdict = Verdict.REVIEW
        reason = (
            f"Aggregate score {agg:.3f} in the {source_value or 'default'} "
            f"review band [{th.allow_below}, {th.block_above})."
        )

    trigger_note = (
        f" Triggered detectors: {', '.join(triggered_names)}."
        if triggered_names
        else ""
    )
    return verdict, float(agg), reason + trigger_note
