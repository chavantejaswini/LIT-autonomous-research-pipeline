"""PubMed E-utilities client.

Endpoints used
--------------
* `esearch.fcgi` — keyword search returning PMIDs. Docs:
  https://www.ncbi.nlm.nih.gov/books/NBK25499/
* `esummary.fcgi` — metadata for a list of PMIDs. Docs:
  https://www.ncbi.nlm.nih.gov/books/NBK25499/

Rate limits
-----------
Without an API key NCBI allows up to **3 requests per second** per IP.
With an `api_key` query parameter the cap rises to **10 req/s**. The
HTTP runner's 429-handling + Retry-After backs off automatically when
the server pushes back.
"""
from __future__ import annotations

import uuid

from ..http import HttpRunner
from ..models import Failure, Success
from ..response_types import (
    PubMedSearchResponse,
    PubMedSummaryResponse,
    parse_json_into,
)
from ._base import call_to_result

SOURCE = "pubmed"
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedClient:
    def __init__(self, runner: HttpRunner, api_key: str | None = None) -> None:
        self._runner = runner
        self._api_key = api_key

    def _common(self, params: dict) -> dict:
        out = dict(params)
        out.setdefault("retmode", "json")
        if self._api_key:
            out["api_key"] = self._api_key
        return out

    def search(self, term: str, retmax: int = 20) -> Success | Failure:
        """`esearch` — return a list of PMIDs matching `term`."""
        qid = f"pubmed.search:{term}:{retmax}:{uuid.uuid4().hex[:6]}"
        return call_to_result(
            self._runner,
            "GET",
            f"{BASE_URL}/esearch.fcgi",
            source=SOURCE,
            query_id=qid,
            params=self._common({"db": "pubmed", "term": term, "retmax": retmax}),
            parser=parse_json_into(PubMedSearchResponse),
        )

    def fetch_by_id(self, pmid: str | int) -> Success | Failure:
        """`esummary` — return article metadata for a single PMID."""
        qid = f"pubmed.fetch:{pmid}:{uuid.uuid4().hex[:6]}"
        return call_to_result(
            self._runner,
            "GET",
            f"{BASE_URL}/esummary.fcgi",
            source=SOURCE,
            query_id=qid,
            params=self._common({"db": "pubmed", "id": str(pmid)}),
            parser=parse_json_into(PubMedSummaryResponse),
        )
