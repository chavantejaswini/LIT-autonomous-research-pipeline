"""FDA FAERS (openFDA drug adverse-event) client.

Endpoints used
--------------
* `GET /drug/event.json` — search drug adverse event reports with a Lucene
  `search` parameter. Docs: https://open.fda.gov/apis/drug/event/
* `GET /drug/event.json?search=safetyreportid:<id>` — fetch a single
  adverse event report by its safety report id.

Rate limits
-----------
Without a key: **240 requests / minute / IP** and **1000 requests / day / IP**.
With a free API key the daily cap rises to **120 000 / day**. 429 responses
include `Retry-After`, which the HTTP runner honors.
"""
from __future__ import annotations

import uuid

from ..http import HttpRunner
from ..models import Failure, Success
from ..response_types import FaersResponse, parse_json_into
from ._base import call_to_result

SOURCE = "faers"
BASE_URL = "https://api.fda.gov/drug/event.json"


class FaersClient:
    def __init__(self, runner: HttpRunner, api_key: str | None = None) -> None:
        self._runner = runner
        self._api_key = api_key

    def _params(self, params: dict) -> dict:
        out = dict(params)
        if self._api_key:
            out["api_key"] = self._api_key
        return out

    def search(self, search_expr: str, limit: int = 10) -> Success | Failure:
        """Run a Lucene `search` expression. Example:
        `patient.drug.medicinalproduct:"METFORMIN"`."""
        qid = f"faers.search:{search_expr}:{limit}:{uuid.uuid4().hex[:6]}"
        return call_to_result(
            self._runner,
            "GET",
            BASE_URL,
            source=SOURCE,
            query_id=qid,
            params=self._params({"search": search_expr, "limit": limit}),
            parser=parse_json_into(FaersResponse),
        )

    def fetch_by_id(self, safety_report_id: str) -> Success | Failure:
        """Fetch one report by its `safetyreportid`."""
        qid = f"faers.fetch:{safety_report_id}:{uuid.uuid4().hex[:6]}"
        return call_to_result(
            self._runner,
            "GET",
            BASE_URL,
            source=SOURCE,
            query_id=qid,
            params=self._params({"search": f"safetyreportid:{safety_report_id}", "limit": 1}),
            parser=parse_json_into(FaersResponse),
        )
