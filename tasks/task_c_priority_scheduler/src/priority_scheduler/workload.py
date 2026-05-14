"""Workload protocol + a deterministic mock implementation for testing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable


class WorkloadError(Exception):
    """Raised by workloads to signal recoverable failure of a callback.

    The scheduler catches *any* exception from `advance`, `checkpoint`, or
    `restore`, but workloads should raise this specific type when the
    failure is expected / handled (e.g. checkpoint disk full).
    """


@runtime_checkable
class Workload(Protocol):
    """The workload tells the scheduler how to checkpoint and resume itself.

    Required attributes (constants for the lifetime of one job):
      * `work_units`     — total ticks of progress to complete.
      * `runtime_cost`   — credits consumed per tick while running.
      * `checkpoint_cost`— credits consumed at each checkpoint.
      * `resume_cost`    — credits consumed at each resume.

    Methods:
      * `advance(ticks)` — perform `ticks` units of work, return new total progress.
      * `checkpoint(path)` — serialize current state to `path`.
      * `restore(path)` — load state from `path`.
      * `is_complete()` — has the workload finished its `work_units`?
    """

    work_units: int
    runtime_cost: int
    checkpoint_cost: int
    resume_cost: int

    def advance(self, ticks: int) -> int: ...
    def checkpoint(self, path: Path) -> None: ...
    def restore(self, path: Path) -> None: ...
    def is_complete(self) -> bool: ...


class MockWorkload:
    """A simple integer-counter workload used by tests and the stress harness.

    Each `advance(ticks)` bumps a `progress` counter by `ticks` (clipped to
    `work_units`). `checkpoint` writes the counter as JSON; `restore` reads
    it back. The mock is intentionally cheap so the stress test runs in
    milliseconds.
    """

    def __init__(
        self,
        name: str,
        work_units: int,
        runtime_cost: int = 1,
        checkpoint_cost: int = 1,
        resume_cost: int = 1,
    ) -> None:
        if work_units < 1:
            raise ValueError("work_units must be ≥ 1")
        self.name = name
        self.work_units = work_units
        self.runtime_cost = runtime_cost
        self.checkpoint_cost = checkpoint_cost
        self.resume_cost = resume_cost
        self.progress = 0
        # Counters for the test harness — verify that no work was lost on resume.
        self.checkpoints_taken = 0
        self.resumes_done = 0

    def advance(self, ticks: int) -> int:
        if ticks < 0:
            raise ValueError("ticks must be ≥ 0")
        self.progress = min(self.work_units, self.progress + ticks)
        return self.progress

    def checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": self.name, "progress": self.progress}))
        self.checkpoints_taken += 1

    def restore(self, path: Path) -> None:
        data = json.loads(path.read_text())
        if data.get("name") != self.name:
            raise ValueError(
                f"checkpoint name mismatch: file={data.get('name')!r}, workload={self.name!r}"
            )
        self.progress = int(data["progress"])
        self.resumes_done += 1

    def is_complete(self) -> bool:
        return self.progress >= self.work_units
