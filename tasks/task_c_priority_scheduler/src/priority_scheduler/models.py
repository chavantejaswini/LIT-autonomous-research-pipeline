"""Data types exposed by the scheduler."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NewType

JobId = NewType("JobId", str)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    PAUSED = "paused"
    RESUMING = "resuming"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"  # workload raised during a callback; scheduler isolated it


# `PriorityLevel` is a logical name. Concrete numeric ranks live in the
# YAML config; the scheduler resolves names to `PriorityTier` objects.
PriorityLevel = NewType("PriorityLevel", str)


@dataclass(frozen=True)
class PriorityTier:
    name: str
    rank: int
    preemptible_by: frozenset[str]


@dataclass(frozen=True)
class JobHandle:
    """Opaque handle returned by `Scheduler.submit`."""

    job_id: JobId
    priority: PriorityLevel


@dataclass
class AllocationSnapshot:
    """Point-in-time view of the scheduler. Returned by `current_allocations`."""

    tick: int
    running: list[JobHandle]
    queued: list[JobHandle]
    paused: list[JobHandle]
    completed: list[JobHandle]
    cancelled: list[JobHandle]
    failed: list[JobHandle]
    credits_used: int
    credits_budget: int

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "running": [(h.job_id, h.priority) for h in self.running],
            "queued": [(h.job_id, h.priority) for h in self.queued],
            "paused": [(h.job_id, h.priority) for h in self.paused],
            "completed": [(h.job_id, h.priority) for h in self.completed],
            "cancelled": [(h.job_id, h.priority) for h in self.cancelled],
            "failed": [(h.job_id, h.priority) for h in self.failed],
            "credits_used": self.credits_used,
            "credits_budget": self.credits_budget,
        }
