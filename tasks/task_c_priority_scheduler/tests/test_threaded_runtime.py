"""Tests for the pluggable `ThreadedRuntime` adapter.

Goals:
  * The scheduler produces identical end-states under SynchronousRuntime
    and ThreadedRuntime — only the per-tick advance step is parallelized.
  * `advance()` actually executes on the pool (wall-clock parallelism).
  * Exceptions raised inside `advance()` running on the pool are routed
    to the scheduler's `FAILED` path, not propagated up.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from priority_scheduler import (
    JobStatus,
    Scheduler,
    SynchronousRuntime,
    ThreadedRuntime,
)
from priority_scheduler.workload import MockWorkload, WorkloadError


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "priorities.yaml"


def _build(runtime, tmp_path) -> Scheduler:
    return Scheduler(
        config_path=CONFIG,
        budget=1_000_000,
        slots=4,
        audit_log_path=tmp_path / "audit.jsonl",
        checkpoint_dir=tmp_path / "ckpts",
        runtime=runtime,
    )


def test_threaded_runtime_produces_same_completions(tmp_path):
    """Same workloads under sync vs threaded → same set of completions."""
    work = [(f"w{i}", 3 + (i % 4)) for i in range(8)]

    def _go(runtime, sub) -> list[str]:
        sched = _build(runtime, sub)
        handles = [
            (sched.submit(MockWorkload(n, work_units=u), "standard"), n)
            for n, u in work
        ]
        sched.run_to_completion()
        sched.shutdown()
        return sorted(
            n for h, n in handles if sched.status(h) == JobStatus.COMPLETED
        )

    sync_done = _go(SynchronousRuntime(), tmp_path / "sync")
    thr_done = _go(ThreadedRuntime(max_workers=4), tmp_path / "thr")
    assert sync_done == thr_done == sorted(n for n, _ in work)


class SlowWorkload(MockWorkload):
    """Sleeps in `advance()` so that wall-clock parallelism is observable."""

    def __init__(self, name: str, sleep_s: float, work_units: int = 1):
        super().__init__(name=name, work_units=work_units)
        self._sleep_s = sleep_s

    def advance(self, ticks: int) -> int:
        time.sleep(self._sleep_s)
        return super().advance(ticks)


def test_threaded_runtime_parallelizes_slow_advances(tmp_path):
    """N workloads each sleeping S seconds should finish in ~S, not N×S, on the pool."""
    n = 4
    sleep_s = 0.15

    runtime = ThreadedRuntime(max_workers=n)
    sched = _build(runtime, tmp_path)
    for i in range(n):
        sched.submit(SlowWorkload(f"slow_{i}", sleep_s=sleep_s, work_units=1), "standard")
    t0 = time.perf_counter()
    sched.run_to_completion()
    elapsed = time.perf_counter() - t0
    sched.shutdown()

    # Sequential would be n*sleep_s ≈ 0.6s; parallel should be well under
    # that. Generous slack to keep the test stable on busy CI.
    assert elapsed < n * sleep_s * 0.6, (
        f"threaded runtime didn't parallelize: elapsed={elapsed:.3f}s, "
        f"sequential lower bound was {n * sleep_s:.3f}s"
    )


class FailingPoolWorkload(MockWorkload):
    def advance(self, ticks: int) -> int:
        raise WorkloadError("synthetic failure inside pool worker")


def test_threaded_runtime_routes_pool_exceptions_to_failed(tmp_path):
    """An exception raised inside a thread-pool worker must be captured and
    turned into FAILED, not silently lost and not propagated."""
    runtime = ThreadedRuntime(max_workers=2)
    sched = _build(runtime, tmp_path)
    bad = FailingPoolWorkload("bad", work_units=5)
    good = MockWorkload("good", work_units=2)
    h_bad = sched.submit(bad, "standard")
    h_good = sched.submit(good, "standard")

    sched.run_to_completion()
    sched.shutdown()

    assert sched.status(h_bad) == JobStatus.FAILED
    assert sched.status(h_good) == JobStatus.COMPLETED


def test_scheduler_context_manager_shuts_down_pool(tmp_path):
    runtime = ThreadedRuntime(max_workers=2)
    with _build(runtime, tmp_path) as sched:
        sched.submit(MockWorkload("a", work_units=2), "standard")
        sched.run_to_completion()
    # If shutdown worked, the pool's threads should be joined.
    assert runtime._pool._shutdown
