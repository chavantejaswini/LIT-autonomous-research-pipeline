"""Unified biomedical evidence client.

Wraps five external sources behind a single Result-typed contract:

  * PubMed (E-utilities)               — `evidence_client.sources.pubmed`
  * ClinicalTrials.gov v2 API           — `evidence_client.sources.clinical_trials`
  * FDA FAERS                           — `evidence_client.sources.faers`
  * NHANES (CDC variable list)          — `evidence_client.sources.nhanes`
  * STRING-DB                           — `evidence_client.sources.string_db`

Every method returns a `Success` or `Failure` — never raises on a network
or 4xx/5xx error. Connection errors, timeouts, JSON-decode failures, parse
failures, and non-2xx responses are all mapped to `Failure`.

Production-ready stack:
  * Retry on 429 + 502/503/504 with `Retry-After` honored, max 3 retries.
  * Per-source `CircuitBreaker` — fail fast for known-dead sources.
  * `InMemoryMetrics` — counts and latency per source, exposed via `.metrics`.
  * Thread-safe `TTLCache` shared across sources.
  * Per-source response schemas — upstream shape changes surface as
    `Failure(message="parse error: …")` instead of `KeyError` deep in caller code.
"""

from .circuit_breaker import CircuitBreaker, CircuitOpenError, State as CircuitState
from .client import EvidenceClient
from .http import RetryPolicy, TTLCache
from .metrics import InMemoryMetrics, MetricsCollector, NullMetrics
from .models import Failure, SearchResult, Success

__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "EvidenceClient",
    "Failure",
    "InMemoryMetrics",
    "MetricsCollector",
    "NullMetrics",
    "RetryPolicy",
    "SearchResult",
    "Success",
    "TTLCache",
]
