"""50-workload stress harness."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .scheduler import Scheduler
from .workload import MockWorkload


PRIORITIES = ["critical", "high", "standard", "opportunistic"]
PRIORITY_WEIGHTS = [0.10, 0.25, 0.40, 0.25]  # bias toward standard-ish loads


@dataclass
class StressResult:
    completed: int
    cancelled: int
    failed: int
    credits_used: int
    credits_budget: int
    total_yields: int
    total_resumes: int
    audit_entries: int
    final_tick: int
    workload_progress: dict[str, int]
    workload_resumes: dict[str, int]
    priority_admission_inversions: int


@dataclass
class SweepResult:
    """Aggregated result of running `run_stress` over many seeds.

    The three invariants are asserted across *every* seed — a single
    violation flips the corresponding `all_*` flag to False.
    """

    seeds_run: int
    all_budgets_respected: bool
    all_priority_invariant_held: bool
    all_yields_match_resumes: bool
    completed_total: int
    failed_total: int
    cancelled_total: int
    total_yields: int
    total_resumes: int
    seed_failures: list[dict]


def build_workloads(n: int, rng: random.Random) -> list[tuple[MockWorkload, str]]:
    out: list[tuple[MockWorkload, str]] = []
    for i in range(n):
        priority = rng.choices(PRIORITIES, weights=PRIORITY_WEIGHTS, k=1)[0]
        work_units = rng.randint(2, 12)
        runtime = rng.randint(1, 3)
        ckpt = rng.randint(1, 3)
        resume = rng.randint(1, 3)
        out.append(
            (
                MockWorkload(
                    name=f"job_{i:03d}",
                    work_units=work_units,
                    runtime_cost=runtime,
                    checkpoint_cost=ckpt,
                    resume_cost=resume,
                ),
                priority,
            )
        )
    return out


def run_stress(
    config_path: str | Path,
    n_workloads: int = 50,
    slots: int = 4,
    seed: int = 7,
    audit_log_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
) -> StressResult:
    rng = random.Random(seed)
    workloads = build_workloads(n_workloads, rng)
    # Budget: generous upper bound — sum of worst-case per-workload cost
    # (runtime + 2 × (checkpoint + resume) for a couple of yields each).
    budget = sum(
        w.work_units * w.runtime_cost + 4 * (w.checkpoint_cost + w.resume_cost)
        for w, _ in workloads
    )

    sched = Scheduler(
        config_path=config_path,
        budget=budget,
        slots=slots,
        audit_log_path=audit_log_path,
        checkpoint_dir=checkpoint_dir,
    )

    handles: list = []
    # Submit workloads in two waves to exercise the pre-emption path.
    # Wave 1 (lower-priority bias): submit at tick 0, run for several ticks.
    # Wave 2 (higher-priority bias): submit while wave 1 is running. The
    # late-arriving criticals/highs preempt any lower-priority running jobs.
    wave1 = workloads[: n_workloads // 2]
    wave2 = workloads[n_workloads // 2 :]
    # Bias wave 2 toward critical/high so preemption is forced.
    wave2 = sorted(wave2, key=lambda x: PRIORITIES.index(x[1]))

    for w, prio in wave1:
        handles.append((sched.submit(w, prio), w))

    # Let wave 1 run for a few ticks before wave 2 arrives.
    for _ in range(3):
        sched.tick()

    for w, prio in wave2:
        handles.append((sched.submit(w, prio), w))

    sched.run_to_completion()

    inversions = _count_admission_inversions(sched, before=False)

    completed = sum(
        1 for h, _ in handles if sched.status(h).value == "completed"
    )
    cancelled = sum(
        1 for h, _ in handles if sched.status(h).value == "cancelled"
    )
    failed = sum(1 for h, _ in handles if sched.status(h).value == "failed")

    audit = sched.audit_log.entries
    yields = sum(1 for e in audit if e.action == "yielded")
    resumes = sum(1 for e in audit if e.action == "resumed")

    snap = sched.current_allocations()
    return StressResult(
        completed=completed,
        cancelled=cancelled,
        failed=failed,
        credits_used=snap.credits_used,
        credits_budget=snap.credits_budget,
        total_yields=yields,
        total_resumes=resumes,
        audit_entries=len(audit),
        final_tick=snap.tick,
        workload_progress={w.name: w.progress for _, w in handles},
        workload_resumes={w.name: w.resumes_done for _, w in handles},
        priority_admission_inversions=inversions,
    )


def run_stress_sweep(
    config_path: str | Path,
    n_workloads: int = 50,
    slots: int = 4,
    seeds: int = 100,
    audit_log_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
) -> SweepResult:
    """Run the stress test under `seeds` different seeds and aggregate.

    Provides much stronger evidence than a single seed: if even one seed
    out of the entire sweep breaks an invariant, the corresponding flag
    in the result is False and the failing seed is recorded.
    """
    audit_dir = Path(audit_log_dir) if audit_log_dir else None
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None

    all_budgets_ok = True
    all_priority_ok = True
    all_yields_resumes_ok = True
    completed_total = 0
    failed_total = 0
    cancelled_total = 0
    yields_total = 0
    resumes_total = 0
    failures: list[dict] = []

    for seed in range(seeds):
        per_seed_audit = (audit_dir / f"audit_seed{seed}.jsonl") if audit_dir else None
        per_seed_ckpt = (ckpt_dir / f"seed{seed}") if ckpt_dir else None
        result = run_stress(
            config_path=config_path,
            n_workloads=n_workloads,
            slots=slots,
            seed=seed,
            audit_log_path=per_seed_audit,
            checkpoint_dir=per_seed_ckpt,
        )
        completed_total += result.completed
        failed_total += result.failed
        cancelled_total += result.cancelled
        yields_total += result.total_yields
        resumes_total += result.total_resumes

        seed_violations = {}
        if result.credits_used > result.credits_budget:
            all_budgets_ok = False
            seed_violations["budget_exceeded"] = (
                result.credits_used, result.credits_budget
            )
        if result.priority_admission_inversions != 0:
            all_priority_ok = False
            seed_violations["priority_inversions"] = result.priority_admission_inversions
        if result.total_yields != result.total_resumes:
            all_yields_resumes_ok = False
            seed_violations["yields_vs_resumes"] = (
                result.total_yields, result.total_resumes
            )
        if seed_violations:
            failures.append({"seed": seed, **seed_violations})

    return SweepResult(
        seeds_run=seeds,
        all_budgets_respected=all_budgets_ok,
        all_priority_invariant_held=all_priority_ok,
        all_yields_match_resumes=all_yields_resumes_ok,
        completed_total=completed_total,
        failed_total=failed_total,
        cancelled_total=cancelled_total,
        total_yields=yields_total,
        total_resumes=resumes_total,
        seed_failures=failures,
    )


def _count_admission_inversions(sched: Scheduler, before: bool) -> int:
    """Replay the audit log: count any `admitted` or `resumed` event where
    a strictly-higher-priority job was still queued or paused at the same tick.

    The scheduler should never admit a lower-priority job while a
    higher-priority job is eligible — this counts violations.
    """
    audit = sched.audit_log.entries
    tier_rank = {
        "critical": 0, "high": 1, "standard": 2, "opportunistic": 3,
    }

    # Build a tick-indexed view of which jobs were queued/paused at each
    # admission decision. We replay forward.
    state: dict[str, tuple[str, str]] = {}  # job_id -> (status, priority)
    inversions = 0
    for e in audit:
        prio = e.priority or ""
        if e.action == "submitted":
            state[e.job_id] = ("queued", prio)
        elif e.action in ("admitted", "resumed"):
            # Anyone in queued/paused with strictly higher priority is an inversion.
            admitted_rank = tier_rank.get(prio, 99)
            for jid, (st, pp) in state.items():
                if jid == e.job_id:
                    continue
                if st in ("queued", "paused") and tier_rank.get(pp, 99) < admitted_rank:
                    inversions += 1
            state[e.job_id] = ("running", prio)
        elif e.action == "yielded":
            state[e.job_id] = ("paused", prio)
        elif e.action == "completed":
            state[e.job_id] = ("completed", prio)
        elif e.action == "cancelled":
            state[e.job_id] = ("cancelled", prio)
    return inversions
