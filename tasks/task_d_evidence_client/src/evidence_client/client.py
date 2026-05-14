"""`EvidenceClient` — one-stop façade that owns the HTTP runner and exposes
each source as a typed attribute.

Production wiring:
  * Per-source `CircuitBreaker` (5 consecutive failures → 30s open).
  * `InMemoryMetrics` for counters + latency, exposed via `.metrics`.
  * Thread-safe TTL cache shared by every source.
"""
from __future__ import annotations

import requests

from .circuit_breaker import CircuitBreaker
from .http import HttpRunner, RetryPolicy, TTLCache
from .metrics import InMemoryMetrics, MetricsCollector
from .sources import (
    ClinicalTrialsClient,
    FaersClient,
    NhanesClient,
    PubMedClient,
    StringDbClient,
)


class EvidenceClient:
    def __init__(
        self,
        *,
        timeout: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        cache: TTLCache | None = None,
        session: requests.Session | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        metrics: MetricsCollector | None = None,
        enable_circuit_breaker: bool = True,
        enable_metrics: bool = True,
        pubmed_api_key: str | None = None,
        faers_api_key: str | None = None,
        string_caller_identity: str = "lit-test/0.1",
    ) -> None:
        # Resolve optional infra. `None` + `enable_*=True` → use a sensible default.
        if circuit_breaker is None and enable_circuit_breaker:
            circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_seconds=30.0)
        if metrics is None and enable_metrics:
            metrics = InMemoryMetrics()

        self._runner = HttpRunner(
            session=session,
            timeout=timeout,
            retry_policy=retry_policy,
            cache=cache,
            circuit_breaker=circuit_breaker,
            metrics=metrics,
        )
        self.pubmed = PubMedClient(self._runner, api_key=pubmed_api_key)
        self.clinical_trials = ClinicalTrialsClient(self._runner)
        self.faers = FaersClient(self._runner, api_key=faers_api_key)
        self.nhanes = NhanesClient(self._runner)
        self.string_db = StringDbClient(self._runner, caller_identity=string_caller_identity)

    @property
    def runner(self) -> HttpRunner:
        return self._runner

    @property
    def metrics(self) -> MetricsCollector:
        return self._runner.metrics

    @property
    def circuit_breaker(self) -> CircuitBreaker | None:
        return self._runner.breaker
