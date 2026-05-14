"""STRING-DB protein-interaction client.

Endpoints used
--------------
* `GET /api/json/network` — protein interaction network for one or more
  identifiers. Docs: https://string-db.org/cgi/help.pl?subpage=api
* `GET /api/json/get_string_ids` — map free-text protein names to STRING
  identifiers (used here as the fetch-by-ID method).

Rate limits
-----------
STRING asks users to delay 1s between calls when running large batches;
no hard 429 in normal usage. The HTTP runner's retry policy applies in
either case.
"""
from __future__ import annotations

import uuid

from ..http import HttpRunner
from ..models import Failure, Success
from ..response_types import StringDbResponse, parse_json_into
from ._base import call_to_result

SOURCE = "string_db"
BASE_URL = "https://string-db.org/api/json"


class StringDbClient:
    def __init__(self, runner: HttpRunner, caller_identity: str = "lit-test/0.1") -> None:
        self._runner = runner
        self._caller = caller_identity

    def search(
        self, identifiers: list[str] | str, species: int = 9606
    ) -> Success | Failure:
        """Fetch the protein-protein interaction network for the given identifiers
        (default species: 9606 = Homo sapiens)."""
        if isinstance(identifiers, list):
            ids = "%0d".join(identifiers)
        else:
            ids = identifiers
        qid = f"string.search:{ids}:{species}:{uuid.uuid4().hex[:6]}"
        return call_to_result(
            self._runner,
            "GET",
            f"{BASE_URL}/network",
            source=SOURCE,
            query_id=qid,
            params={
                "identifiers": ids,
                "species": species,
                "caller_identity": self._caller,
            },
            parser=parse_json_into(StringDbResponse),
        )

    def fetch_by_id(self, name: str, species: int = 9606) -> Success | Failure:
        """Map a single protein name to its STRING identifier(s)."""
        qid = f"string.fetch:{name}:{species}:{uuid.uuid4().hex[:6]}"
        return call_to_result(
            self._runner,
            "GET",
            f"{BASE_URL}/get_string_ids",
            source=SOURCE,
            query_id=qid,
            params={
                "identifiers": name,
                "species": species,
                "caller_identity": self._caller,
            },
        )
