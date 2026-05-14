"""ClinicalTrials.gov v2 API client.

Endpoints used
--------------
* `GET /api/v2/studies` — list/search studies. Docs:
  https://clinicaltrials.gov/api/v2/studies
* `GET /api/v2/studies/{NCTID}` — fetch one study by NCT id. Docs:
  https://clinicaltrials.gov/data-api/api

Rate limits
-----------
The v2 API has no documented hard limit; CTG asks users to be
"reasonable." The HTTP runner enforces 429-Retry-After regardless.
"""
from __future__ import annotations

import uuid

from ..http import HttpRunner
from ..models import Failure, Success
from ..response_types import CtgSearchResponse, CtgStudyResponse, parse_json_into
from ._base import call_to_result

SOURCE = "clinical_trials"
BASE_URL = "https://clinicaltrials.gov/api/v2"


class ClinicalTrialsClient:
    def __init__(self, runner: HttpRunner) -> None:
        self._runner = runner

    def search(self, query: str, page_size: int = 10) -> Success | Failure:
        """Free-text search across study fields."""
        qid = f"ctg.search:{query}:{page_size}:{uuid.uuid4().hex[:6]}"
        return call_to_result(
            self._runner,
            "GET",
            f"{BASE_URL}/studies",
            source=SOURCE,
            query_id=qid,
            params={
                "query.term": query,
                "pageSize": page_size,
                "format": "json",
            },
            parser=parse_json_into(CtgSearchResponse),
        )

    def fetch_by_id(self, nct_id: str) -> Success | Failure:
        """Fetch one study record by its NCT identifier."""
        qid = f"ctg.fetch:{nct_id}:{uuid.uuid4().hex[:6]}"
        return call_to_result(
            self._runner,
            "GET",
            f"{BASE_URL}/studies/{nct_id}",
            source=SOURCE,
            query_id=qid,
            params={"format": "json"},
            parser=parse_json_into(CtgStudyResponse),
        )
