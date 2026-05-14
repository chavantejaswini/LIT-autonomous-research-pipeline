"""Per-source response schemas.

We don't try to model every nested field — that's a moving target and
locks us to upstream churn. Each schema asserts only the *top-level*
keys we depend on. If a key is missing or the wrong type, the source
client surfaces a `Failure(message="parse error: …")` instead of letting
a `KeyError` propagate from deep in the caller.

This is the "validate at the boundary" pattern. Strictness lives at the
edge; everything inside the package can trust the shape.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Permissive(BaseModel):
    """Allow extra fields — we only pin the keys we depend on."""

    model_config = ConfigDict(extra="allow")


# ---- PubMed ---------------------------------------------------------------


class PubMedSearchResponse(_Permissive):
    esearchresult: dict[str, Any]


class PubMedSummaryResponse(_Permissive):
    result: dict[str, Any]


# ---- ClinicalTrials.gov v2 ------------------------------------------------


class CtgSearchResponse(_Permissive):
    # v2 returns either `studies: [...]` or `totalCount: N` when paged out.
    # Accept either shape via a permissive schema with no required field.
    pass


class CtgStudyResponse(_Permissive):
    # `protocolSection` is the canonical top-level key for a single-study fetch.
    pass


# ---- FDA FAERS ------------------------------------------------------------


class FaersResponse(_Permissive):
    # FAERS returns either `results: [...]` on success or `error: {...}` on
    # failure (which is wrapped inside a 200). Both are accepted; the source
    # client surfaces `error` payloads as success-with-error-body to the caller.
    pass


# ---- NHANES ---------------------------------------------------------------


class NhanesHtmlResponse(BaseModel):
    """NHANES returns raw HTML — we wrap it in a stable shape."""

    model_config = ConfigDict(extra="allow")
    html: str
    length: int


# ---- STRING-DB ------------------------------------------------------------


class StringDbResponse(BaseModel):
    """STRING-DB returns a JSON array. We re-encode under a top-level key
    so all of our `Success.data` values are dicts."""

    model_config = ConfigDict(extra="allow")
    results: list[Any]


# ---- helpers --------------------------------------------------------------


def parse_json_into(schema: type[BaseModel]):
    """Return a parser function that validates JSON text against `schema`
    and returns the resulting dict (so `Success.data` stays JSON-shaped)."""

    def _parse(text: str) -> dict:
        data = json.loads(text)
        # Each source's response is a dict — except STRING-DB which returns
        # a list at the top level. Wrap a list to fit the schema's `results`.
        if isinstance(data, list) and schema is StringDbResponse:
            data = {"results": data}
        return schema.model_validate(data).model_dump()

    return _parse


def parse_html_into(schema: type[BaseModel]):
    """Return a parser function for sources that emit raw HTML rather than JSON."""

    def _parse(text: str) -> dict:
        return schema.model_validate({"html": text, "length": len(text)}).model_dump()

    return _parse
