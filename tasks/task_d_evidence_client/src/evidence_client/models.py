"""Result types for the evidence client.

`SearchResult` is a discriminated union of `Success` and `Failure`. The
discriminator field is `status` — Pydantic will route deserialization
correctly when the union is used in a parent model.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class Success(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["success"] = "success"
    data: list[Any] | dict[str, Any]
    source: str
    fetched_at: datetime
    query_id: str

    def ok(self) -> bool:  # convenience for "if result.ok(): ..."
        return True


class Failure(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["failure"] = "failure"
    source: str
    status_code: int | None
    message: str
    fetched_at: datetime
    query_id: str

    def ok(self) -> bool:
        return False


SearchResult = Union[Success, Failure]


# Helpers used by the source modules to build results consistently.

def make_success(
    source: str, data: Any, query_id: str
) -> Success:
    from datetime import timezone

    return Success(
        data=data,
        source=source,
        fetched_at=datetime.now(timezone.utc),
        query_id=query_id,
    )


def make_failure(
    source: str, status_code: int | None, message: str, query_id: str
) -> Failure:
    from datetime import timezone

    return Failure(
        source=source,
        status_code=status_code,
        message=message,
        fetched_at=datetime.now(timezone.utc),
        query_id=query_id,
    )
