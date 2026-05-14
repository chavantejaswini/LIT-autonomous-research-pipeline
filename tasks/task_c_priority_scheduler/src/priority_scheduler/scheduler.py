"""The Scheduler.

Discrete-time model:
  * `submit(workload, priority)` enqueues a job.
  * `tick()` advances simulated time by one unit: handle admissions and
    pre-emptions, advance running jobs, account credits.
  * `run_to_completion()` ticks until every job is in a terminal state
    or the budget is exhausted.

Pre-emption rule: when a higher-priority job is queued and all slots are
occupied by jobs that it can pre-empt, the lowest-priority preemptible
running job yields, checkpoints to disk, and re-enters the queue.

Production hardening:
  * Workload callbacks (`advance`, `checkpoint`, `restore`) are wrapped in
    try/except. A misbehaving workload transitions to FAILED — the
    scheduler itself never crashes from foreign-code errors.
  * Checkpoint files are deleted once their job reaches a terminal state.
  * The `_admit` / `_resume` paths enforce a real-time priority guard:
    they refuse to put a job in a slot if a strictly-higher-priority
    pending job is eligible for that slot.
  * The per-tick work-advance step is delegated to a pluggable
    `RuntimeAdapter` — the default is synchronous, but `ThreadedRuntime`
    runs the advances of independent jobs on a thread pool.
"""
from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from .audit import AuditLog
from .models import (
    AllocationSnapshot,
    JobHandle,
    JobId,
    JobStatus,
    PriorityLevel,
    PriorityTier,
)
from .runtime import RuntimeAdapter, SynchronousRuntime
from .workload import Workload, WorkloadError


@dataclass
class _Job:
    handle: JobHandle
    workload: Workload
    tier: PriorityTier
    status: JobStatus = JobStatus.QUEUED
    submit_seq: int = 0  # tie-breaker for FIFO within a priority tier
    checkpoint_path: Path | None = None

    @property
    def is_running(self) -> bool:
        return self.status == JobStatus.RUNNING

    @property
    def is_paused(self) -> bool:
        return self.status == JobStatus.PAUSED


_TERMINAL_STATES = (JobStatus.COMPLETED, JobStatus.CANCELLED, JobStatus.FAILED)


class BudgetExceededError(RuntimeError):
    pass


class UnknownJobError(KeyError):
    pass


class PriorityInversionError(RuntimeError):
    """Raised if the real-time priority guard detects an attempted inversion.

    This is a defensive invariant — it should never fire under correct
    scheduling code. Surfacing it loudly makes regressions trivial to spot.
    """


class Scheduler:
    def __init__(
        self,
        config_path: str | Path,
        budget: int,
        slots: int = 4,
        audit_log_path: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        runtime: RuntimeAdapter | None = None,
    ) -> None:
        if slots < 1:
            raise ValueError("slots must be ≥ 1")
        self._slots = slots
        self._budget = int(budget)
        self._credits_used = 0
        self._tick = 0
        self._submit_counter = itertools.count()

        raw = yaml.safe_load(Path(config_path).read_text())
        self._tiers: dict[str, PriorityTier] = {}
        for t in raw["tiers"]:
            self._tiers[t["name"]] = PriorityTier(
                name=t["name"],
                rank=int(t["rank"]),
                preemptible_by=frozenset(t.get("preemptible_by", []) or []),
            )

        self._jobs: dict[JobId, _Job] = {}
        self._audit = AuditLog(audit_log_path)
        self._checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else Path("./checkpoints")
        )
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._runtime = runtime if runtime is not None else SynchronousRuntime()

    # ---- public API -------------------------------------------------------

    def submit(self, workload: Workload, priority: PriorityLevel) -> JobHandle:
        if priority not in self._tiers:
            raise ValueError(
                f"unknown priority tier: {priority!r} (known: {list(self._tiers)})"
            )
        jid = JobId(uuid.uuid4().hex[:12])
        handle = JobHandle(job_id=jid, priority=priority)
        job = _Job(
            handle=handle,
            workload=workload,
            tier=self._tiers[priority],
            status=JobStatus.QUEUED,
            submit_seq=next(self._submit_counter),
        )
        self._jobs[jid] = job
        self._audit.record(
            tick=self._tick,
            action="submitted",
            job_id=jid,
            priority=priority,
            reason="submit() called",
        )
        return handle

    def cancel(self, handle: JobHandle) -> None:
        job = self._get(handle)
        if job.status in _TERMINAL_STATES:
            return
        prev_status = job.status
        job.status = JobStatus.CANCELLED
        self._cleanup_checkpoint(job)
        self._audit.record(
            tick=self._tick,
            action="cancelled",
            job_id=handle.job_id,
            priority=handle.priority,
            reason=f"cancel() called from {prev_status.value}",
        )

    def status(self, handle: JobHandle) -> JobStatus:
        return self._get(handle).status

    def current_allocations(self) -> AllocationSnapshot:
        by_status: dict[JobStatus, list[JobHandle]] = {s: [] for s in JobStatus}
        for j in self._jobs.values():
            by_status[j.status].append(j.handle)
        return AllocationSnapshot(
            tick=self._tick,
            running=by_status[JobStatus.RUNNING],
            queued=by_status[JobStatus.QUEUED],
            paused=by_status[JobStatus.PAUSED],
            completed=by_status[JobStatus.COMPLETED],
            cancelled=by_status[JobStatus.CANCELLED],
            failed=by_status[JobStatus.FAILED],
            credits_used=self._credits_used,
            credits_budget=self._budget,
        )

    @property
    def audit_log(self) -> AuditLog:
        return self._audit

    def shutdown(self) -> None:
        """Release runtime resources (thread pool, etc.). Idempotent."""
        self._runtime.shutdown()

    def __enter__(self) -> "Scheduler":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()

    def tick(self) -> None:
        """Advance the scheduler by one unit of time.

        Order matters:
          1. Fill free slots with the highest-priority pending jobs across
             both paused and queued sets — priority strictly dominates
             FIFO-among-paused.
          2. If queued/paused jobs out-rank running jobs, pre-empt to make room.
          3. Advance every running job by 1 work unit (delegated to runtime).
        """
        self._tick += 1

        self._fill_slots()
        self._preempt_if_needed()
        self._advance_running()

    def run_to_completion(self, max_ticks: int = 100_000) -> None:
        for _ in range(max_ticks):
            if self._all_terminal():
                return
            self.tick()
        raise RuntimeError(
            f"scheduler did not converge within {max_ticks} ticks — "
            f"likely a budget or deadlock issue"
        )

    # ---- internals --------------------------------------------------------

    def _get(self, handle: JobHandle) -> _Job:
        job = self._jobs.get(handle.job_id)
        if job is None:
            raise UnknownJobError(handle.job_id)
        return job

    def _all_terminal(self) -> bool:
        return all(j.status in _TERMINAL_STATES for j in self._jobs.values())

    def _running(self) -> list[_Job]:
        return [j for j in self._jobs.values() if j.is_running]

    def _paused(self) -> list[_Job]:
        return [j for j in self._jobs.values() if j.is_paused]

    def _queued(self) -> list[_Job]:
        return [j for j in self._jobs.values() if j.status == JobStatus.QUEUED]

    def _by_priority(self, jobs: list[_Job]) -> list[_Job]:
        """Sort by (rank ascending, submit_seq ascending) — head is highest priority."""
        return sorted(jobs, key=lambda j: (j.tier.rank, j.submit_seq))

    def _free_slots(self) -> int:
        return self._slots - len(self._running())

    def _fill_slots(self) -> None:
        """Fill free slots by priority across paused + queued combined."""
        if self._free_slots() <= 0:
            return
        candidates = self._by_priority(self._paused() + self._queued())
        for job in candidates:
            if self._free_slots() <= 0:
                return
            if job.is_paused:
                self._resume(job, reason="slot available")
            else:
                self._admit(job, reason="slot available")

    def _preempt_if_needed(self) -> None:
        """Drive a pre-emption when a queued job can displace a running job.

        We do at most one preemption per pending arrival per tick to avoid
        thrash; multiple high-priority arrivals get processed across ticks.
        """
        for queued_job in self._by_priority(self._queued() + self._paused()):
            target = self._find_preemption_target(queued_job)
            if target is None:
                # If the highest-priority pending can't preempt anything,
                # lower-priority pending certainly can't either — stop.
                return
            ok = self._preempt(
                target,
                reason=f"yielding to {queued_job.tier.name}/{queued_job.handle.job_id}",
            )
            if not ok:
                # Target failed to checkpoint and is now FAILED, which frees
                # its slot anyway. Move to the next queued candidate next tick.
                continue
            # After yielding, the target's slot is free; admit the new job.
            if queued_job.is_paused:
                self._resume(queued_job, reason="resumed after pre-empting lower-priority")
            else:
                self._admit(queued_job, reason="admitted after pre-empting lower-priority")

    def _find_preemption_target(self, candidate: _Job) -> _Job | None:
        """Return the lowest-priority running job that `candidate` may pre-empt."""
        victims = [
            r
            for r in self._running()
            if candidate.tier.name in r.tier.preemptible_by
        ]
        if not victims:
            return None
        # Lowest priority among victims (highest rank), then most-recently submitted.
        return sorted(victims, key=lambda j: (-j.tier.rank, -j.submit_seq))[0]

    def _admit(self, job: _Job, reason: str) -> None:
        # Real-time priority guard: refuse to admit if a strictly-higher-priority
        # job is currently pending (queued or paused). This is defense in depth
        # on top of the priority-ordered fill in `_fill_slots`.
        self._enforce_no_priority_inversion(job)
        job.status = JobStatus.RUNNING
        self._audit.record(
            tick=self._tick,
            action="admitted",
            job_id=job.handle.job_id,
            priority=job.handle.priority,
            reason=reason,
        )

    def _resume(self, job: _Job, reason: str) -> None:
        self._enforce_no_priority_inversion(job)
        if job.checkpoint_path is None:
            raise RuntimeError(
                f"job {job.handle.job_id} has no checkpoint to restore from"
            )
        self._charge(job.workload.resume_cost, "resume")
        job.status = JobStatus.RESUMING
        try:
            job.workload.restore(job.checkpoint_path)
        except (WorkloadError, Exception) as exc:
            self._mark_failed(job, f"restore failed: {exc!r}")
            return
        job.status = JobStatus.RUNNING
        self._audit.record(
            tick=self._tick,
            action="resumed",
            job_id=job.handle.job_id,
            priority=job.handle.priority,
            reason=reason,
            credits_charged=job.workload.resume_cost,
        )

    def _preempt(self, job: _Job, reason: str) -> bool:
        """Drive one pre-emption; returns True on success, False if the
        workload failed during its checkpoint callback."""
        self._charge(job.workload.checkpoint_cost, "checkpoint")
        job.status = JobStatus.CHECKPOINTING
        path = self._checkpoint_dir / f"{job.handle.job_id}.json"
        try:
            job.workload.checkpoint(path)
        except (WorkloadError, Exception) as exc:
            self._mark_failed(job, f"checkpoint failed: {exc!r}")
            return False
        job.checkpoint_path = path
        job.status = JobStatus.PAUSED
        self._audit.record(
            tick=self._tick,
            action="yielded",
            job_id=job.handle.job_id,
            priority=job.handle.priority,
            reason=reason,
            credits_charged=job.workload.checkpoint_cost,
        )
        return True

    def _advance_running(self) -> None:
        """Charge runtime cost, then ask the runtime adapter to advance the
        workloads. The adapter may run them sequentially or in parallel."""
        running_now = list(self._running())
        for job in running_now:
            self._charge(job.workload.runtime_cost, f"runtime/{job.handle.job_id}")

        # The adapter returns a dict mapping job_id → exception (or None).
        # Wrapping foreign-code execution this way keeps the scheduler itself
        # immune to workload bugs.
        outcomes = self._runtime.advance_all(running_now, ticks=1)

        for job in running_now:
            err = outcomes.get(job.handle.job_id)
            if err is not None:
                self._mark_failed(job, f"advance failed: {err!r}")
                continue
            try:
                done = job.workload.is_complete()
            except Exception as exc:
                self._mark_failed(job, f"is_complete failed: {exc!r}")
                continue
            if done:
                job.status = JobStatus.COMPLETED
                self._cleanup_checkpoint(job)
                self._audit.record(
                    tick=self._tick,
                    action="completed",
                    job_id=job.handle.job_id,
                    priority=job.handle.priority,
                    reason="workload reported is_complete=True",
                )

    def _enforce_no_priority_inversion(self, job: _Job) -> None:
        """Refuse to give `job` a slot if a strictly-higher-priority pending
        job is eligible. Raises PriorityInversionError — a bug in the
        scheduler's own logic, not a user-facing error."""
        for other in self._queued() + self._paused():
            if other.handle.job_id == job.handle.job_id:
                continue
            if other.tier.rank < job.tier.rank:
                raise PriorityInversionError(
                    f"attempted to admit {job.tier.name}/{job.handle.job_id} "
                    f"while higher-priority {other.tier.name}/{other.handle.job_id} "
                    "was still pending"
                )

    def _mark_failed(self, job: _Job, reason: str) -> None:
        job.status = JobStatus.FAILED
        self._cleanup_checkpoint(job)
        self._audit.record(
            tick=self._tick,
            action="failed",
            job_id=job.handle.job_id,
            priority=job.handle.priority,
            reason=reason,
        )

    def _cleanup_checkpoint(self, job: _Job) -> None:
        if job.checkpoint_path is None:
            return
        try:
            job.checkpoint_path.unlink(missing_ok=True)
        except OSError:
            # Best-effort: a stuck checkpoint file is housekeeping debt,
            # not a scheduler correctness issue.
            pass
        job.checkpoint_path = None

    def _charge(self, credits: int, reason: str) -> None:
        if credits < 0:
            raise ValueError("credit cost must be non-negative")
        if self._credits_used + credits > self._budget:
            raise BudgetExceededError(
                f"would exceed budget: used={self._credits_used} + {credits} "
                f"> budget={self._budget} ({reason})"
            )
        self._credits_used += credits
