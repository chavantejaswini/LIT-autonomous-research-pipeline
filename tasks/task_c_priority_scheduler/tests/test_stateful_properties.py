"""Stateful property testing of the scheduler with `hypothesis`.

Where the existing tests are example-based ("submit these three workloads,
check this specific outcome"), this test generates *random sequences* of
submit / cancel / tick operations and verifies that several scheduler
invariants hold at every single intermediate step — plus a few global
invariants checked once the scheduler is drained.

When hypothesis finds a sequence that violates an invariant, it
automatically *shrinks* the failing sequence to the smallest one that
still reproduces the bug, so failures are minimal and reproducible.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, settings, strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule

from priority_scheduler import Scheduler
from priority_scheduler.workload import MockWorkload

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "priorities.yaml"
PRIORITIES = ["critical", "high", "standard", "opportunistic"]


class SchedulerStateMachine(RuleBasedStateMachine):
    """Random-walk state machine for the scheduler.

    Per-step invariants (checked after every rule):
      * `credits_used <= budget` — never overspend.
      * `len(running) <= slots` — never overpack.
      * Every submitted job is in exactly one terminal-or-active state —
        the lifecycle accounting never drifts.

    Global invariants (checked in `teardown` after draining to completion):
      * `yields == resumes` — no pre-empted job is silently abandoned.
      * Every non-cancelled job reaches `COMPLETED`.
    """

    handles = Bundle("handles")

    def __init__(self) -> None:
        super().__init__()
        self.slots = 3
        # Budget is intentionally huge so we exercise scheduling logic, not
        # budget-exhaustion edge cases (those have their own targeted tests).
        self.budget = 10_000_000
        self.tmpdir = Path(tempfile.mkdtemp(prefix="scheduler_hyp_"))
        self.sched = Scheduler(
            config_path=CONFIG,
            budget=self.budget,
            slots=self.slots,
            audit_log_path=self.tmpdir / "audit.jsonl",
            checkpoint_dir=self.tmpdir / "ckpts",
        )
        self.submit_count = 0
        self.workload_counter = 0

    # ---- rules: random operations the scheduler should survive -------------

    @rule(
        target=handles,
        priority=st.sampled_from(PRIORITIES),
        units=st.integers(min_value=1, max_value=8),
        runtime_cost=st.integers(min_value=1, max_value=3),
        ckpt_cost=st.integers(min_value=1, max_value=2),
        resume_cost=st.integers(min_value=1, max_value=2),
    )
    def submit_job(self, priority, units, runtime_cost, ckpt_cost, resume_cost):
        self.workload_counter += 1
        w = MockWorkload(
            name=f"w_{self.workload_counter}",
            work_units=units,
            runtime_cost=runtime_cost,
            checkpoint_cost=ckpt_cost,
            resume_cost=resume_cost,
        )
        h = self.sched.submit(w, priority)
        self.submit_count += 1
        return h

    @rule()
    def advance_one_tick(self):
        self.sched.tick()

    @rule(handle=handles)
    def cancel_a_job(self, handle):
        # Cancel is documented as idempotent on terminal jobs, so this is
        # safe to fire on any handle the state machine has already produced.
        self.sched.cancel(handle)

    # ---- per-step invariants -----------------------------------------------

    @invariant()
    def credits_within_budget(self):
        snap = self.sched.current_allocations()
        assert snap.credits_used <= snap.credits_budget, (
            f"credits_used={snap.credits_used} > budget={snap.credits_budget}"
        )

    @invariant()
    def running_count_within_slots(self):
        snap = self.sched.current_allocations()
        assert len(snap.running) <= self.slots, (
            f"running={len(snap.running)} > slots={self.slots}"
        )

    @invariant()
    def lifecycle_accounting_is_consistent(self):
        snap = self.sched.current_allocations()
        total = (
            len(snap.running)
            + len(snap.queued)
            + len(snap.paused)
            + len(snap.completed)
            + len(snap.cancelled)
            + len(snap.failed)
        )
        assert total == self.submit_count, (
            f"accounting drift: in-buckets={total}, submitted={self.submit_count}"
        )

    @invariant()
    def each_job_appears_exactly_once(self):
        """A job must never appear in two state buckets simultaneously."""
        snap = self.sched.current_allocations()
        all_ids = [
            h.job_id
            for bucket in (
                snap.running, snap.queued, snap.paused,
                snap.completed, snap.cancelled, snap.failed,
            )
            for h in bucket
        ]
        assert len(all_ids) == len(set(all_ids)), (
            f"duplicate job_id across buckets: {all_ids}"
        )

    # ---- global invariants checked after draining --------------------------

    def teardown(self):
        try:
            self.sched.run_to_completion()
        finally:
            # 1. After draining, no job should be left in a transient state.
            snap = self.sched.current_allocations()
            assert len(snap.running) == 0, f"still running after drain: {snap.running}"
            assert len(snap.queued) == 0, f"still queued after drain: {snap.queued}"
            assert len(snap.paused) == 0, f"still paused after drain: {snap.paused}"

            # 2. Every submitted job must terminate.
            audit = self.sched.audit_log.entries
            submits = sum(1 for e in audit if e.action == "submitted")
            cancels = sum(1 for e in audit if e.action == "cancelled")
            completes = sum(1 for e in audit if e.action == "completed")
            fails = sum(1 for e in audit if e.action == "failed")
            assert submits == cancels + completes + fails, (
                f"submit={submits} cancel={cancels} complete={completes} "
                f"fail={fails} — some jobs reached no terminal state"
            )

            # 3. Per-job: every job that yielded must subsequently either
            # resume (normal flow) or be cancelled (user choice). It must
            # not vanish in a paused state.
            per_job: dict[str, list[str]] = {}
            for e in audit:
                if e.job_id is not None:
                    per_job.setdefault(e.job_id, []).append(e.action)
            for jid, actions in per_job.items():
                while "yielded" in actions:
                    idx = actions.index("yielded")
                    after = actions[idx + 1:]
                    assert "resumed" in after or "cancelled" in after, (
                        f"job {jid} yielded but never resumed or was cancelled: {actions}"
                    )
                    # Strip one yield-then-resume/cancel pair and recheck.
                    if "resumed" in after:
                        actions = actions[: idx] + after[after.index("resumed") + 1:]
                    else:
                        break  # cancelled is terminal — no more yields possible

            shutil.rmtree(self.tmpdir, ignore_errors=True)


# Wrap the state machine into a pytest TestCase and tune the search budget.
# `suppress_health_check` quiets a warning about the per-test setup cost
# of constructing a Scheduler.
SchedulerStateMachine.TestCase.settings = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
TestSchedulerStateful = SchedulerStateMachine.TestCase
