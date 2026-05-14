"""Pre-emption + checkpoint/resume behavior."""
from __future__ import annotations

from priority_scheduler import JobStatus
from priority_scheduler.workload import MockWorkload


def test_critical_preempts_standard(scheduler) -> None:
    # Slots = 2. Fill them with standards, then submit a critical: it must
    # preempt one of the standards.
    w1 = MockWorkload("a", work_units=10)
    w2 = MockWorkload("b", work_units=10)
    h1 = scheduler.submit(w1, "standard")
    h2 = scheduler.submit(w2, "standard")
    scheduler.tick()
    assert {scheduler.status(h1), scheduler.status(h2)} == {JobStatus.RUNNING}

    w_crit = MockWorkload("crit", work_units=3)
    h_crit = scheduler.submit(w_crit, "critical")
    scheduler.tick()  # admit critical, preempt one standard
    snap = scheduler.current_allocations()
    assert h_crit in snap.running
    # Exactly one of (h1, h2) must now be paused with a checkpoint.
    paused_ids = {h.job_id for h in snap.paused}
    assert len(paused_ids) == 1
    paused = h1 if h1.job_id in paused_ids else h2
    # And the audit log records the yield.
    actions = scheduler.audit_log.actions_for(paused.job_id)
    assert "yielded" in actions


def test_critical_is_non_preemptible(scheduler) -> None:
    """A critical job cannot be pre-empted by anything."""
    w_crit = MockWorkload("crit", work_units=5)
    w_crit2 = MockWorkload("crit2", work_units=5)
    h1 = scheduler.submit(w_crit, "critical")
    h2 = scheduler.submit(w_crit2, "critical")
    scheduler.tick()  # both critical, fill both slots
    snap = scheduler.current_allocations()
    assert len(snap.running) == 2
    # A third critical arrives. It cannot pre-empt either of the running
    # ones (criticals are not preemptible), so it must wait.
    w3 = MockWorkload("crit3", work_units=2)
    h3 = scheduler.submit(w3, "critical")
    scheduler.tick()
    assert scheduler.status(h3) == JobStatus.QUEUED


def test_high_does_not_preempt_high(scheduler) -> None:
    """`high` is only preemptible by `critical`."""
    w1 = MockWorkload("h1", work_units=5)
    w2 = MockWorkload("h2", work_units=5)
    scheduler.submit(w1, "high")
    scheduler.submit(w2, "high")
    scheduler.tick()
    snap = scheduler.current_allocations()
    assert len(snap.running) == 2

    w3 = MockWorkload("h3", work_units=2)
    h3 = scheduler.submit(w3, "high")
    scheduler.tick()
    # Slots full of `high`; new `high` cannot preempt → stays queued.
    assert scheduler.status(h3) == JobStatus.QUEUED


def test_checkpoint_resume_preserves_progress(scheduler) -> None:
    """A pre-empted workload resumes from where it left off, not from scratch."""
    w_std = MockWorkload("std", work_units=10)
    h_std = scheduler.submit(w_std, "standard")
    w_op = MockWorkload("op", work_units=10)
    scheduler.submit(w_op, "opportunistic")
    # Tick a few times to make progress on both.
    for _ in range(3):
        scheduler.tick()
    progress_before_yield = w_std.progress
    assert progress_before_yield > 0

    # Now a critical arrives — preempts the lowest-priority running job (op),
    # then continues. Standard keeps running.
    w_crit = MockWorkload("crit", work_units=2)
    scheduler.submit(w_crit, "critical")
    scheduler.tick()
    scheduler.run_to_completion()
    assert w_op.is_complete()
    assert w_op.resumes_done >= 1
    assert w_std.is_complete()


def test_opportunistic_preempted_by_higher(scheduler) -> None:
    """Opportunistic is the easiest to displace — anything can preempt it."""
    w_op = MockWorkload("op", work_units=20)
    w_op2 = MockWorkload("op2", work_units=20)
    scheduler.submit(w_op, "opportunistic")
    scheduler.submit(w_op2, "opportunistic")
    scheduler.tick()
    # Submit a standard — must preempt one opportunistic.
    w_std = MockWorkload("std", work_units=2)
    h_std = scheduler.submit(w_std, "standard")
    scheduler.tick()
    snap = scheduler.current_allocations()
    assert h_std in snap.running
    paused_count = len(snap.paused)
    assert paused_count == 1
