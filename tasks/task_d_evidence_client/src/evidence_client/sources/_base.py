"""Shared helper for source clients: convert `HttpRunner` outcomes to `Result`s.

Also propagates the `source` label down to the runner so that per-source
circuit breaker + metrics can attribute calls correctly.
"""
from __future__ import annotations

import json
from typing import Any

from ..circuit_breaker import CircuitOpenError
from ..http import HttpResponse, HttpRunner
from ..models import Failure, Success, make_failure, make_success


def call_to_result(
    runner: HttpRunner,
    method: str,
    url: str,
    *,
    source: str,
    query_id: str,
    params: dict | None = None,
    body: Any = None,
    headers: dict | None = None,
    parser: callable | None = None,
) -> Success | Failure:
    """Run an HTTP call and wrap the outcome in `Success` / `Failure`.

    `parser` is applied to the raw response text — the default is JSON.
    Any parser exception is captured and surfaced as a `Failure`.

    The `source` argument is propagated to the runner so the breaker and
    metrics layer can attribute the call correctly.
    """
    if method.upper() == "GET":
        resp = runner.get(url, source=source, params=params, headers=headers)
    elif method.upper() == "POST":
        resp = runner.post(
            url, source=source, params=params, json_body=body, headers=headers
        )
    else:
        return make_failure(source, None, f"unsupported method: {method}", query_id)

    if isinstance(resp, CircuitOpenError):
        return make_failure(source, None, "circuit-open: source temporarily disabled", query_id)
    if isinstance(resp, RuntimeError):
        return make_failure(source, None, f"network error: {resp}", query_id)
    if not isinstance(resp, HttpResponse):
        return make_failure(
            source, None, f"unexpected response object: {type(resp).__name__}", query_id
        )

    if not (200 <= resp.status_code < 300):
        snippet = resp.text[:512]
        return make_failure(
            source, resp.status_code,
            f"HTTP {resp.status_code}: {snippet}",
            query_id,
        )

    try:
        parsed = parser(resp.text) if parser is not None else json.loads(resp.text)
    except Exception as exc:
        return make_failure(
            source, resp.status_code, f"parse error: {exc!r}", query_id
        )

    return make_success(source, parsed, query_id)
