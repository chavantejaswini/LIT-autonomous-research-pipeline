"""Append-only audit log of allocation changes."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditEntry:
    timestamp: str
    tick: int
    action: str
    job_id: str | None
    priority: str | None
    reason: str
    extra: dict[str, Any] = field(default_factory=dict)


class AuditLog:
    """Append-only log mirrored both in-memory and to a JSONL file.

    Reads are O(n). The on-disk log is the source of truth for replay; the
    in-memory mirror is for the scheduler's runtime introspection and tests.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._entries: list[AuditEntry] = []
        self._path: Path | None = Path(path) if path is not None else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Reset on init so stress runs start clean. Truncate explicitly
            # rather than appending to ambiguous prior contents.
            self._path.write_text("")

    def record(
        self,
        tick: int,
        action: str,
        job_id: str | None = None,
        priority: str | None = None,
        reason: str = "",
        **extra: Any,
    ) -> AuditEntry:
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            tick=tick,
            action=action,
            job_id=job_id,
            priority=priority,
            reason=reason,
            extra=dict(extra),
        )
        self._entries.append(entry)
        if self._path is not None:
            with self._path.open("a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        return entry

    @property
    def entries(self) -> list[AuditEntry]:
        """Return a copy — entries themselves are frozen."""
        return list(self._entries)

    def actions_for(self, job_id: str) -> list[str]:
        return [e.action for e in self._entries if e.job_id == job_id]
