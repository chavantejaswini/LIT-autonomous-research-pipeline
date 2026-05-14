# Task D — External Evidence Client Library

A unified Python client that wraps five external biomedical evidence
sources behind a single `Result`-typed contract. **No method raises on
a network or 4xx/5xx error** — every outcome surfaces as `Success` or
`Failure` so callers can handle them uniformly.

**Production-hardened:** retries 429 + transient 5xx with `Retry-After`,
fails fast on dead sources via per-source circuit breakers, emits
structured metrics, validates upstream responses with per-source schemas,
and is safe to call concurrently from many threads.

## Install & run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Replay-only: tests run against recorded cassettes, no network needed.
pytest -v

# Re-record cassettes against the real APIs.
VCR_RECORD=1 pytest -v

# Run the live integration tests (gated; one per source).
EVIDENCE_LIVE=1 pytest -v -k live
```

## API at a glance

```python
from evidence_client import EvidenceClient

c = EvidenceClient(timeout=30.0)

r = c.pubmed.search("CRISPR aging", retmax=10)
if r.ok():
    print(r.data["esearchresult"]["idlist"])
else:
    print(f"failed: {r.status_code} {r.message}")

# Observability — built-in metrics
print(c.metrics.snapshot())
# {'pubmed': {'calls_by_outcome': {'ok': 1}, 'calls_by_status': {'200': 1},
#             'latency_ms': {'count': 1, 'mean': 124.3, ...}}}

# Circuit breaker state
print(c.circuit_breaker.snapshot())
# {'pubmed': {'state': 'closed', 'consecutive_failures': 0}}
```

### Result types

```python
class Success(BaseModel):
    status: Literal["success"]
    data: list | dict
    source: str
    fetched_at: datetime
    query_id: str

class Failure(BaseModel):
    status: Literal["failure"]
    source: str
    status_code: int | None     # None for pure network / circuit-open failures
    message: str
    fetched_at: datetime
    query_id: str

SearchResult = Union[Success, Failure]
```

## Sources

| Source | Module | Search method | Fetch-by-ID method | Endpoint | Rate limit |
|--------|--------|---------------|--------------------|----------|------------|
| PubMed | `evidence_client.sources.pubmed` | `search(term, retmax)` | `fetch_by_id(pmid)` | `eutils.ncbi.nlm.nih.gov/entrez/eutils/{esearch,esummary}.fcgi` | 3 req/s without key; 10 req/s with `api_key` |
| ClinicalTrials.gov v2 | `sources.clinical_trials` | `search(query, page_size)` | `fetch_by_id(nct_id)` | `clinicaltrials.gov/api/v2/studies[/{NCT}]` | "Be reasonable" — no documented hard cap |
| FDA FAERS | `sources.faers` | `search(lucene_expr, limit)` | `fetch_by_id(safetyreportid)` | `api.fda.gov/drug/event.json` | 240/min, 1000/day without key; 120k/day with key |
| NHANES (CDC) | `sources.nhanes` | `search(component, cycle_year)` | `fetch_by_id(cycle, file_name)` | `wwwn.cdc.gov/Nchs/Nhanes/Search/variablelist.aspx` and per-cycle pages | No published cap; runner backs off on 429 |
| STRING-DB | `sources.string_db` | `search(identifiers, species)` | `fetch_by_id(name, species)` | `string-db.org/api/json/{network,get_string_ids}` | Advisory 1s gap between calls in batches |

## Production-readiness stack

### 1. Retry on 429 + transient 5xx

`RetryPolicy(max_retries=3, retry_on_status={429, 502, 503, 504})`. The runner:
- Honors `Retry-After` (both delta-seconds and HTTP-date forms).
- Falls back to exponential backoff (`base_delay × 2^attempt`, capped at `max_delay`).
- After exhausting retries on 5xx, surfaces the response as a `Failure(status_code=…)` so the caller can decide.
- Network errors are retried up to `max_retries` and then surface as `Failure(message="network error: …")`.

500 ("internal server error" — server bug, not transient) is **not** retried by default. Callers can extend the retry set via `RetryPolicy(retry_on_status=…)`.

### 2. Per-source `CircuitBreaker`

`CircuitBreaker(failure_threshold=5, recovery_seconds=30.0)` per source:

- **CLOSED** — calls flow through normally.
- **OPEN** — after 5 consecutive failures, calls fail fast as `Failure(message="circuit-open: …")` without touching the network.
- **HALF_OPEN** — after `recovery_seconds`, the next call is allowed through as a probe. Success closes the circuit; failure re-opens it for another window.

This protects the institute from wasting time + credits on a known-dead upstream. Each source has its own state; if PubMed is down, FAERS still works.

### 3. Structured metrics — `MetricsCollector`

`InMemoryMetrics` aggregates per source:

- `calls_by_outcome` — counts keyed by `ok | http_error | network_error | circuit_open | cache_hit`.
- `calls_by_status` — counts keyed by HTTP status code.
- `latency_ms` — count, sum, min, max, mean.

The default is `NullMetrics` (zero cost). Plug in `OpenTelemetryMetrics` or similar by implementing the `MetricsCollector` protocol — the runner doesn't care.

### 4. Thread-safe `TTLCache`

Internal `threading.Lock` protects every `get/set/evict`. Tested with 8 concurrent worker threads issuing 20 calls each — no deadlocks, no lost entries.

The circuit breaker is also thread-safe (50 concurrent failure-recordings still converge to OPEN exactly once).

### 5. Per-source response schemas

Each source has a pydantic schema (`PubMedSearchResponse`, `CtgSearchResponse`, etc.) that pins only the **top-level** keys the client depends on. If the upstream API changes shape, the call surfaces as:

```python
Failure(message="parse error: 1 validation error for PubMedSearchResponse\n  esearchresult: Field required")
```

instead of a `KeyError` deep in a downstream agent.

## VCR cassettes

Cassettes live in `tests/cassettes/` — one YAML per test function. Each
cassette stores the recorded request + response, scrubbed of `api_key`
query params and `authorization`/`cookie` headers.

Default `pytest` runs in **replay-only** (`record_mode="none"`) — any
HTTP call without a matching cassette fails the test. To re-record:
`VCR_RECORD=1 pytest` switches to `new_episodes`. Always commit the
regenerated cassettes.

## Live integration tests

One `@live_only()` test per source. They run **only** when
`EVIDENCE_LIVE=1` is set. They assert only the contract (`r.ok()` or
well-formed `Failure`) — never pin specific records that could change.

## Tests

**47 passing + 5 live-gated** across nine test modules:

| Test module | What it covers |
|---|---|
| `test_http_layer.py` (11) | Cache hit/miss/expiry, 2xx-only caching, 429 retry, network-error path, parse-error path, `Retry-After` parsing |
| `test_5xx_retry.py` (4) | 502/503 retry-then-succeed, 504 give-up, 500 not retried by default, configurable retry set |
| `test_circuit_breaker.py` (8) | Threshold trip, per-source isolation, success resets counter, HALF_OPEN probe, HALF_OPEN re-open on failure, runner-level short-circuit, 5xx-driven breaker open, `Failure` surface |
| `test_metrics.py` (6) | Successful call counts, HTTP error vs ok distinction, network-error counts, cache-hit attribution, per-source attribution, `EvidenceClient.metrics` exposure |
| `test_thread_safety.py` (3) | Concurrent cache `get/set` safe, runner thread-safe under 8-thread load, breaker converges to OPEN under concurrent failures |
| `test_typed_responses.py` (4) | Schema accepts valid payloads, rejects missing top-level keys, STRING-DB list-unwrap, malformed upstream → `Failure` |
| `test_pubmed/clinical_trials/faers/nhanes/string_db.py` (10) | Replay tests for search + fetch-by-id per source |
| `*_live` (5, gated) | One real-API integration test per source |
