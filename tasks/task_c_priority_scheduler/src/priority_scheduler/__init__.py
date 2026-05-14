"""Compute-resource priority scheduler with pre-emption and checkpoint/resume.

The scheduler operates in discrete `tick`s and maintains a fixed number
of execution slots. At each tick it:

  1. Admits the highest-priority queued job whenever a slot is free.
  2. Pre-empts a lower-priority running job when a higher-priority job
     arrives and all slots are full. The pre-empted job checkpoints to
     disk and re-enters the queue at the head of its priority tier.
  3. Advances each running job by one work unit, debiting credits from
     the shared budget.
  4. Resumes paused jobs from their checkpoint when a slot opens up.

Every allocation change is appended to an immutable audit log.

Production hardening:
  * Workload callbacks (`advance`, `checkpoint`, `restore`) are isolated
    in try/except; failures route to the `FAILED` terminal state.
  * Checkpoint files are cleaned up when their job reaches a terminal state.
  * Real-time `PriorityInversionError` guard on admission/resume.
  * Pluggable `RuntimeAdapter` (`SynchronousRuntime` default,
    `ThreadedRuntime` for real-thread parallelism).
"""

from .audit import AuditEntry, AuditLog
from .models import (
    AllocationSnapshot,
    JobHandle,
    JobStatus,
    PriorityLevel,
    PriorityTier,
)
from .runtime import RuntimeAdapter, SynchronousRuntime, ThreadedRuntime
from .scheduler import (
    BudgetExceededError,
    PriorityInversionError,
    Scheduler,
    UnknownJobError,
)
from .workload import Workload, WorkloadError

__all__ = [
    "AllocationSnapshot",
    "AuditEntry",
    "AuditLog",
    "BudgetExceededError",
    "JobHandle",
    "JobStatus",
    "PriorityInversionError",
    "PriorityLevel",
    "PriorityTier",
    "RuntimeAdapter",
    "Scheduler",
    "SynchronousRuntime",
    "ThreadedRuntime",
    "UnknownJobError",
    "Workload",
    "WorkloadError",
]
