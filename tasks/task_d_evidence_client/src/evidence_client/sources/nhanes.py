"""NHANES (CDC) variable & dataset client.

NHANES does not expose a JSON REST API — it publishes per-cycle HTML
variable lists and binary `.xpt` data files. We surface a thin wrapper
around two stable HTML endpoints. Callers parse the HTML themselves
(the raw HTML is returned in `Success.data["html"]`).

Endpoints used
--------------
* `GET /Nchs/Nhanes/Search/variablelist.aspx?Component=<component>&CycleBeginYear=<year>` —
  variable listing for a survey cycle.
  Docs: https://wwwn.cdc.gov/Nchs/Nhanes/Search/default.aspx
* `GET /Nchs/Nhanes/<cycle>/<file>` — dataset file metadata page.

Rate limits
-----------
CDC does not publish a numeric limit; their hosting infrastructure has
historically rate-limited bursts. The HTTP runner caps at 3 retries with
exponential backoff on any 429.
"""
from __future__ import annotations

import uuid

from ..http import HttpRunner
from ..models import Failure, Success
from ..response_types import NhanesHtmlResponse, parse_html_into
from ._base import call_to_result

SOURCE = "nhanes"
BASE_URL = "https://wwwn.cdc.gov/Nchs/Nhanes"

_parse_nhanes = parse_html_into(NhanesHtmlResponse)


class NhanesClient:
    def __init__(self, runner: HttpRunner) -> None:
        self._runner = runner

    def search(self, component: str, cycle_year: int) -> Success | Failure:
        """List variables for a given component (Demographics/Examination/Laboratory/...)
        and cycle. Returns the raw HTML payload."""
        qid = f"nhanes.search:{component}:{cycle_year}:{uuid.uuid4().hex[:6]}"
        return call_to_result(
            self._runner,
            "GET",
            f"{BASE_URL}/Search/variablelist.aspx",
            source=SOURCE,
            query_id=qid,
            params={"Component": component, "CycleBeginYear": cycle_year},
            parser=_parse_nhanes,
        )

    def fetch_by_id(self, cycle: str, file_name: str) -> Success | Failure:
        """Fetch a single dataset metadata page, e.g. `cycle="2017-2018"`,
        `file_name="DEMO_J.htm"`."""
        qid = f"nhanes.fetch:{cycle}/{file_name}:{uuid.uuid4().hex[:6]}"
        return call_to_result(
            self._runner,
            "GET",
            f"{BASE_URL}/{cycle}/{file_name}",
            source=SOURCE,
            query_id=qid,
            parser=_parse_nhanes,
        )
