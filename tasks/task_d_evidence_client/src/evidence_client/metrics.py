"""Metrics collection for the evidence client.

The default `NullMetrics` is a no-op; production deployments can plug in
their own collector that mirrors counts and latency into Prometheus /
OpenTelemetry / etc. `InMemoryMetrics` is provided for tests and local
observability.

Recorded dimensions per HTTP call:
  * `source`     — "pubmed", "faers", ...
  * `outcome`    — "ok", "http_error", "network_error", "circuit_open", "cache_hit"
  * `status_code`— int or None
  * `latency_ms` — float
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Protocol


class MetricsCollector(Protocol):
    def record(
        self,
        *,
        source: str,
        status_code: int | None,
        latency_ms: float,
        outcome: str,
    ) -> None: ...

    def snapshot(self) -> dict: ...


class NullMetrics:
    """No-op collector. The default — costs nothing when metrics aren't wanted."""

    def record(self, **kwargs) -> None:
        pass

    def snapshot(self) -> dict:
        return {}


class InMemoryMetrics:
    """Thread-safe in-memory metrics for tests and local introspection.

    Aggregates per source:
      * `calls_total{outcome=…}` — counts
      * `status_total{status_code=…}` — counts (omits None)
      * latency: count, sum, min, max, mean (p50/p99 require a real
        histogram backend in prod — this is for ops-quality summaries)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str], int] = defaultdict(int)
        self._statuses: dict[tuple[str, int], int] = defaultdict(int)
        self._lat_count: dict[str, int] = defaultdict(int)
        self._lat_sum: dict[str, float] = defaultdict(float)
        self._lat_min: dict[str, float] = {}
        self._lat_max: dict[str, float] = {}

    def record(
        self,
        *,
        source: str,
        status_code: int | None,
        latency_ms: float,
        outcome: str,
    ) -> None:
        with self._lock:
            self._counts[(source, outcome)] += 1
            if status_code is not None:
                self._statuses[(source, status_code)] += 1
            if latency_ms > 0.0:
                self._lat_count[source] += 1
                self._lat_sum[source] += latency_ms
                if source not in self._lat_min or latency_ms < self._lat_min[source]:
                    self._lat_min[source] = latency_ms
                if source not in self._lat_max or latency_ms > self._lat_max[source]:
                    self._lat_max[source] = latency_ms

    def snapshot(self) -> dict:
        with self._lock:
            sources = (
                {s for (s, _) in self._counts.keys()}
                | {s for (s, _) in self._statuses.keys()}
                | set(self._lat_count)
            )
            out: dict = {}
            for src in sorted(sources):
                lat_n = self._lat_count.get(src, 0)
                out[src] = {
                    "calls_by_outcome": {
                        o: c for (s, o), c in self._counts.items() if s == src
                    },
                    "calls_by_status": {
                        str(code): c for (s, code), c in self._statuses.items() if s == src
                    },
                    "latency_ms": {
                        "count": lat_n,
                        "mean": (self._lat_sum[src] / lat_n) if lat_n else 0.0,
                        "min": self._lat_min.get(src, 0.0),
                        "max": self._lat_max.get(src, 0.0),
                    },
                }
            return out
