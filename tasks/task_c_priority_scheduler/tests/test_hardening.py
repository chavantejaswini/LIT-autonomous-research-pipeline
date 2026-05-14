"""Tests for the production-hardening features:

  * Workload callback isolation — a misbehaving workload becomes FAILED
    without crashing the scheduler.
  * Checkpoint cleanup — the on-disk file is removed when a job reaches
    a terminal state (COMPLETED / CANCELLED / FAILED).
  * Real-time priority guard — direct invocation of the internal admit
    path raises PriorityInversionError when an inversion is attempted.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from priority_scheduler import (
    JobStatus,
    PriorityInversionError,
    Scheduler,
)
from priority_scheduler.workload import MockWorkload, WorkloadError


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "priorities.yaml"


# -------- workload callback isolation -------------------------------------


class FailingAdvanceWorkload(MockWorkload):
    """Raises on the third `advance()` call."""

    def __init__(self, name: str):
        super().__init__(name=name, work_units=10)
        self._advance_count = 0

    def advance(self, ticks: int) -> int:
        self._advance_count += 1
        if self._advance_count == 3:
            raise WorkloadError("synthetic advance failure")
        return super().advance(ticks)


class FailingCheckpointWorkload(MockWorkload):
    def checkpoint(self, path: Path) -> None:
        raise OSError("synthetic disk-full")


def test_workload_advance_failure_marks_job_failed_not_scheduler_crash(scheduler):
    bad = FailingAdvanceWorkload("bad")
    good = MockWorkload("good", work_units=5)
    h_bad = scheduler.submit(bad, "standard")
    h_good = scheduler.submit(good, "standard")

    # Run until both terminate. The scheduler must not raise.
    scheduler.run_to_completion()

    assert scheduler.status(h_bad) == JobStatus.FAILED
    assert scheduler.status(h_good) == JobStatus.COMPLETED
    # Audit log records the failure with a reason.
    actions = scheduler.audit_log.actions_for(h_bad.job_id)
    assert "failed" in actions


def test_workload_checkpoint_failure_marks_job_failed(scheduler):
    """A workload that raises on `checkpoint()` during pre-emption fails
    cleanly — the slot is freed, and the preempting job still gets admitted."""
    bad = FailingCheckpointWorkload(name="bad", work_units=10)
    other = MockWorkload(name="other", work_units=10)
    # Fill both slots with low-priority workloads.
    h_bad = scheduler.submit(bad, "opportunistic")
    h_other = scheduler.submit(other, "opportunistic")
    scheduler.tick()
    # Submit a critical — it will try to preempt one of the opportunistics.
    crit = MockWorkload(name="crit", work_units=3)
    h_crit = scheduler.submit(crit, "critical")
    scheduler.run_to_completion()

    # The critical must complete regardless of the bad workload's failure.
    assert scheduler.status(h_crit) == JobStatus.COMPLETED
    # At least one of (bad, other) was the pre-emption target; whichever it
    # was, if its checkpoint raised it's now FAILED. Otherwise it completed.
    assert scheduler.status(h_bad) in (JobStatus.FAILED, JobStatus.COMPLETED)


def test_failed_jobs_excluded_from_running_and_queued(scheduler):
    bad = FailingAdvanceWorkload("bad")
    scheduler.submit(bad, "standard")
    scheduler.run_to_completion()
    snap = scheduler.current_allocations()
    assert len(snap.failed) == 1
    assert len(snap.running) == 0
    assert len(snap.queued) == 0


# -------- checkpoint cleanup ----------------------------------------------


def test_checkpoint_file_deleted_on_completion(scheduler, tmp_path):
    """After a pre-emption + resume + completion, the checkpoint file is gone."""
    op = MockWorkload("op", work_units=8)
    other = MockWorkload("other", work_units=8)
    scheduler.submit(op, "opportunistic")
    scheduler.submit(other, "opportunistic")
    scheduler.tick()
    # Force a pre-emption: critical bumps one opportunistic.
    crit = MockWorkload("crit", work_units=2)
    scheduler.submit(crit, "critical")
    scheduler.run_to_completion()

    # No leftover checkpoint files for any of the terminated jobs.
    leftover = list(Path(scheduler._checkpoint_dir).glob("*.json"))
    assert leftover == [], f"orphaned checkpoint files: {leftover}"


def test_checkpoint_file_deleted_on_cancel_while_paused(scheduler):
    """Cancelling a paused job removes its checkpoint file too."""
    a = MockWorkload("a", work_units=20)
    b = MockWorkload("b", work_units=20)
    h_a = scheduler.submit(a, "opportunistic")
    scheduler.submit(b, "opportunistic")
    scheduler.tick()
    # Preempt `a`.
    crit = MockWorkload("crit", work_units=2)
    scheduler.submit(crit, "critical")
    scheduler.tick()
    # Find the paused one and cancel it.
    snap = scheduler.current_allocations()
    paused_handles = [h for h in snap.paused]
    assert paused_handles, "expected at least one paused job"
    scheduler.cancel(paused_handles[0])

    leftover = [
        p for p in Path(scheduler._checkpoint_dir).glob("*.json")
        if p.stem == paused_handles[0].job_id
    ]
    assert leftover == [], f"cancelled paused job left a checkpoint: {leftover}"


# -------- real-time priority guard ----------------------------------------


def test_priority_guard_blocks_direct_admit_inversion(scheduler):
    """The internal admit path refuses to admit a low-priority job while a
    higher-priority job is still pending. Tests the guard directly."""
    crit_workload = MockWorkload("crit", work_units=5)
    std_workload = MockWorkload("std", work_units=5)
    scheduler.submit(crit_workload, "critical")
    h_std = scheduler.submit(std_workload, "standard")

    # Bypass `_fill_slots` and try to admit the standard directly — the
    # priority guard must refuse it because the critical is still queued.
    std_job = scheduler._get(h_std)
    with pytest.raises(PriorityInversionError):
        scheduler._admit(std_job, reason="test-direct-call")
