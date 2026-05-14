# Task C — Compute Resource Priority Scheduler with Pre-emption

A discrete-time scheduler that allocates execution slots to ML / scientific
workloads under a four-tier priority order, with clean checkpoint-and-resume
on pre-emption and an append-only audit log of every allocation change.

**Production-hardened:** isolates misbehaving workloads, cleans up checkpoint
files on terminal states, defends against priority inversion in real time,
and ships a pluggable runtime adapter so the same scheduler logic can drive
either deterministic simulation or a real OS thread pool.

## Install & run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Single-seed stress test (writes results/stress_report.json).
scheduler-stress

# Multi-seed sweep — runs 100 seeds and asserts all three invariants
# hold across every one (writes results/stress_sweep_report.json).
scheduler-stress-sweep

pytest -v
```

## API at a glance

```python
from priority_scheduler import (
    Scheduler, SynchronousRuntime, ThreadedRuntime,
)
from priority_scheduler.workload import MockWorkload  # or your own

# Default: synchronous runtime — deterministic and easy to test.
sched = Scheduler(
    config_path="configs/priorities.yaml",
    budget=10_000,
    slots=4,
)

h = sched.submit(MockWorkload("trial_007", work_units=20), "high")
sched.status(h)              # JobStatus.QUEUED
sched.current_allocations()  # AllocationSnapshot
sched.run_to_completion()

# Real-thread parallelism: same logic, different runtime.
with Scheduler(..., runtime=ThreadedRuntime(max_workers=4)) as sched:
    sched.submit(my_io_heavy_workload, "standard")
    sched.run_to_completion()
```

### `Workload` protocol

Callers implement four pieces:

```python
class Workload(Protocol):
    work_units: int        # total ticks of progress
    runtime_cost: int      # credits per tick
    checkpoint_cost: int   # credits per checkpoint
    resume_cost: int       # credits per resume

    def advance(self, ticks: int) -> int: ...
    def checkpoint(self, path: Path) -> None: ...
    def restore(self, path: Path) -> None: ...
    def is_complete(self) -> bool: ...
```

Workloads can raise `WorkloadError` (or any `Exception`) from any callback.
The scheduler catches it, transitions the job to `FAILED`, records the
reason in the audit log, and keeps running.

## Priority tiers

Configured in [`configs/priorities.yaml`](configs/priorities.yaml):

| Name | Rank | Preemptible by |
|---|---|---|
| `critical` | 0 | nothing — non-preemptible |
| `high` | 1 | `critical` |
| `standard` | 2 | `critical`, `high` |
| `opportunistic` | 3 | `critical`, `high`, `standard` |

## Tick semantics

Each `tick()` performs, in order:

1. **Fill free slots.** Highest-priority pending job (across paused +
   queued, by rank then submit order) takes any empty slot. Paused jobs
   resume from their on-disk checkpoint.
2. **Pre-empt if needed.** If a pending job out-ranks a running job and
   may pre-empt it (per the YAML's `preemptible_by`), the lowest-priority
   eligible running job yields: checkpoints to disk, releases its slot,
   re-enters the queue at the head of its tier.
3. **Advance.** Every running job consumes `runtime_cost` credits.
   The per-tick work-advance is delegated to the configured
   `RuntimeAdapter` (sync by default; threaded available).

The scheduler's bookkeeping (admission decisions, audit-log writes) is
single-threaded and deterministic regardless of runtime — only the work
inside `workload.advance()` may run in parallel.

## Production hardening

### 1. Workload callback isolation
Every call into `workload.advance()`, `.checkpoint()`, `.restore()`, and
`.is_complete()` is wrapped in `try/except`. A failure routes the job
through `_mark_failed` → `JobStatus.FAILED` with the reason in the audit
log. The scheduler itself never crashes from foreign-code errors.

### 2. Checkpoint cleanup
Once a job reaches `COMPLETED`, `CANCELLED`, or `FAILED`, its checkpoint
file (if any) is removed. No orphan files left on disk.

### 3. Real-time priority guard
`_admit` and `_resume` both call `_enforce_no_priority_inversion`. If
admission code ever attempted to place a lower-priority job into a slot
while a strictly-higher-priority job remained pending,
`PriorityInversionError` is raised — a defensive invariant on top of the
priority-ordered fill logic.

### 4. Pluggable runtime adapter
[`runtime.py`](src/priority_scheduler/runtime.py):

- `SynchronousRuntime` — default. Advances each running workload one at
  a time. Deterministic and reproducible.
- `ThreadedRuntime(max_workers=N)` — advances each running workload in
  parallel on a `ThreadPoolExecutor`. Pool exceptions are captured and
  routed to `FAILED` just like synchronous failures.

Scheduler decisions are identical across runtimes; only the per-tick
work execution differs.

## Audit log

JSONL on disk + in-memory mirror. Action types:
`submitted`, `admitted`, `yielded`, `resumed`, `completed`, `cancelled`,
`failed`. Each entry carries `tick`, `timestamp`, `job_id`, `priority`,
`reason`, and any extra metadata. Append-only — the scheduler never
seeks or rewrites.

## Stress results

### Single seed (`scheduler-stress --seed 42`)

```
50 workloads · 4 slots · two-wave submission

completed:                          50 / 50
credits_used / budget:              725 / 1521      ← (a) budget respected
total yields / resumes:             1 / 1           ← (c) yield matches resume
priority_admission_inversions:      0                ← (b) priority order respected
```

### Multi-seed sweep (`scheduler-stress-sweep --seeds 100`)

```
100 seeds × 50 workloads = 5000 jobs total · 108 pre-emptions

all_budgets_respected:              true            ← (a)
all_priority_invariant_held:        true            ← (b)
all_yields_match_resumes:           true            ← (c)
completed_total:                    5000 / 5000
seed_failures:                      []
```

## Tests

**23 tests** across the suite:

- `test_scheduler_basic.py` (5) — submit, run, cancel, audit, snapshot.
- `test_preemption.py` (5) — every pre-emption rule from the brief.
- `test_hardening.py` (6) — workload-failure isolation, checkpoint cleanup, real-time priority guard.
- `test_threaded_runtime.py` (4) — same end-state under sync vs threaded, wall-clock parallelism, pool-exception routing, context-manager shutdown.
- `test_stress.py` (1) — single-seed 50-workload stress.
- `test_stress_sweep.py` (1) — **100-seed sweep, asserts every invariant holds across all 5000 jobs**.
- `test_stateful_properties.py` (1) — **`hypothesis`-driven random submit/cancel/tick sequences over 100 examples**, with per-step + global invariants.

Combined, these exercise the scheduler over ~30,000+ randomized operations per test run.
