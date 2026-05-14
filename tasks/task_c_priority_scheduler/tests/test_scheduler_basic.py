"""Smoke tests for the scheduler."""
from __future__ import annotations

from priority_scheduler import JobStatus
from priority_scheduler.workload import MockWorkload


def test_single_workload_runs_to_completion(scheduler) -> None:
    w = MockWorkload("a", work_units=3)
    h = scheduler.submit(w, "standard")
    assert scheduler.status(h) == JobStatus.QUEUED
    scheduler.run_to_completion()
    assert scheduler.status(h) == JobStatus.COMPLETED
    assert w.progress == 3


def test_two_workloads_share_slots(scheduler) -> None:
    w1, w2 = MockWorkload("a", 3), MockWorkload("b", 3)
    h1 = scheduler.submit(w1, "standard")
    h2 = scheduler.submit(w2, "standard")
    scheduler.tick()  # admits both into the 2 slots
    alloc = scheduler.current_allocations()
    assert len(alloc.running) == 2
    scheduler.run_to_completion()
    assert scheduler.status(h1) == JobStatus.COMPLETED
    assert scheduler.status(h2) == JobStatus.COMPLETED


def test_cancel_pending_job(scheduler) -> None:
    w = MockWorkload("a", 100)
    h = scheduler.submit(w, "opportunistic")
    scheduler.cancel(h)
    assert scheduler.status(h) == JobStatus.CANCELLED
    scheduler.run_to_completion()
    assert scheduler.status(h) == JobStatus.CANCELLED


def test_audit_log_records_submit(scheduler) -> None:
    w = MockWorkload("a", 1)
    h = scheduler.submit(w, "standard")
    actions = scheduler.audit_log.actions_for(h.job_id)
    assert actions == ["submitted"]


def test_current_allocations_snapshot_shape(scheduler) -> None:
    w = MockWorkload("a", 3)
    scheduler.submit(w, "standard")
    snap = scheduler.current_allocations()
    assert snap.tick == 0
    assert snap.credits_budget == 100_000
    assert snap.credits_used == 0
    assert len(snap.queued) == 1
